from GLFC import GLFC_model
from ResNet import resnet18_cbam
import torch
import copy
import random
import os.path as osp
import os
from myNetwork import network, LeNet
from Fed_utils import * 
from ProxyServer import * 
from mini_imagenet import *
from tiny_imagenet import *
from option import args_parser
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
import datetime
from multiprocessing import freeze_support

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


def resolve_data_root(path):
    if not path:
        return ''
    if os.path.isfile(path):
        return os.path.dirname(path)
    if os.path.isdir(path):
        for candidate in [path, os.path.join(path, 'federated_data'), os.path.join(path, 'data')]:
            if os.path.isdir(candidate):
                client_files = [f for f in os.listdir(candidate) if f.startswith('client_') and f.endswith('.pt')]
                if client_files:
                    return candidate
        return path
    return path


def resolve_test_path(path):
    if not path:
        return ''
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for candidate in ['global_test_data.pt', 'test_data.pt']:
            full_path = os.path.join(path, candidate)
            if os.path.exists(full_path):
                return full_path
        for candidate_dir in [path, os.path.join(path, 'federated_data'), os.path.join(path, 'data')]:
            for candidate in ['global_test_data.pt', 'test_data.pt']:
                full_path = os.path.join(candidate_dir, candidate)
                if os.path.exists(full_path):
                    return full_path
        return os.path.join(path, 'global_test_data.pt')
    return path


def resolve_kaggle_dataset_paths(args):
    if args.data_root:
        data_root = resolve_data_root(args.data_root)
    else:
        data_root = ''

    if args.test_path:
        test_path = resolve_test_path(args.test_path)
    else:
        test_path = ''

    if data_root and test_path:
        return data_root, test_path

    if args.client_dataset == '100':
        candidates = [
            '/kaggle/input/datasets/npngn123/data-iot-100-client/CICIoT_label_skew_100_clients_bounded_nested_original_order_from_pt/federated_data',
            '/kaggle/input/datasets/npngn123/data-iot-100-client/CICIoT_label_skew_100_clients_bounded_nested_original_order_from_pt',
        ]
    elif args.client_dataset == '200':
        candidates = [
            '/kaggle/input/datasets/npngn123/data-iot-200-client/CICIoT_label_skew_200_clients_bounded_nested_original_order_from_pt/federated_data',
            '/kaggle/input/datasets/npngn123/data-iot-200-client/CICIoT_label_skew_200_clients_bounded_nested_original_order_from_pt',
        ]
    else:
        candidates = [
            '/kaggle/input/datasets/npngn123/data-iot-100-client/CICIoT_label_skew_100_clients_bounded_nested_original_order_from_pt/federated_data',
            '/kaggle/input/datasets/npngn123/data-iot-100-client/CICIoT_label_skew_100_clients_bounded_nested_original_order_from_pt',
            '/kaggle/input/datasets/npngn123/data-iot-200-client/CICIoT_label_skew_200_clients_bounded_nested_original_order_from_pt/federated_data',
            '/kaggle/input/datasets/npngn123/data-iot-200-client/CICIoT_label_skew_200_clients_bounded_nested_original_order_from_pt',
            '/kaggle/input/datasets/npngn123/glfc-data3',
            '/kaggle/input/datasets/npngn123/glfc-data',
            '/kaggle/input/glfc-data',
            '/kaggle/input',
        ]

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        if os.path.isdir(candidate):
            client_files = [f for f in os.listdir(candidate) if f.startswith('client_') and f.endswith('.pt')]
            if client_files:
                data_root = candidate
                test_path = ''
                for test_candidate in [
                    os.path.join(candidate, 'global_test_data.pt'),
                    os.path.join(os.path.dirname(candidate), 'global_test_data.pt'),
                    os.path.join(candidate, 'test_data.pt'),
                ]:
                    if os.path.exists(test_candidate):
                        test_path = test_candidate
                        break
                if not test_path:
                    test_path = os.path.join(os.path.dirname(candidate), 'global_test_data.pt')
                return data_root, test_path

    return data_root or '/kaggle/input/datasets/npngn123/data-iot-100-client/CICIoT_label_skew_100_clients_bounded_nested_original_order_from_pt/federated_data', test_path or '/kaggle/input/datasets/npngn123/data-iot-100-client/CICIoT_label_skew_100_clients_bounded_nested_original_order_from_pt/global_test_data.pt'


def main():
    args = args_parser()

    ## parameters for learning
    if args.dataset == 'tabular':
        from FederatedTabularDataset import FederatedTabularDataset
        from myNetwork import MLP_FeatureExtractor, MLP_Encoder, CNN_FeatureExtractor, CNN_Encoder
        
        # Thiết lập giá trị mặc định "thông minh" cho Tabular nếu người dùng không truyền tham số khác
        if args.num_clients is None:
            if args.client_dataset == '200':
                args.num_clients = 200
            else:
                args.num_clients = 100
        elif args.client_dataset in {'100', '200'} and args.num_clients != int(args.client_dataset):
            print(f"[INFO] Đã override num_clients từ {args.num_clients} thành {int(args.client_dataset)} để khớp với dataset {args.client_dataset} client")
            args.num_clients = int(args.client_dataset)
        if args.tasks_global == 10:
            args.tasks_global = 6
        if args.epochs_global == 100:
            args.epochs_global = 36
        if args.epochs_local == 20:
            args.epochs_local = 5
        if args.local_clients > args.num_clients:
            args.local_clients = args.num_clients
        
        args.task_size    = 6
        args.numclass     = args.task_size
        args.learning_rate = 0.01   # LR nhỏ hơn phù hợp với MLP/tabular/CNN
        args.batch_size = 4096      # Tăng batch size theo yêu cầu của người dùng

        # Cấu hình đường dẫn cho Kaggle nếu bật flag --kaggle (hoặc tự động phát hiện)
        if args.kaggle:
            print("[INFO] Đang chạy trong môi trường Kaggle. Tự động cấu hình đường dẫn.")
            
            args.data_root, args.test_path = resolve_kaggle_dataset_paths(args)
            args.log_base  = '/kaggle/working/training_log'
            args.checkpoint_dir = '/kaggle/working/checkpoints'
            print(f"[INFO] Đã chọn dữ liệu tại: {args.data_root}")
            print(f"[INFO] Đã chọn file test tại: {args.test_path}")
        else:
            if args.data_root:
                args.data_root = resolve_data_root(args.data_root)
            else:
                args.data_root = 'federated_data_final'
            if args.test_path:
                args.test_path = resolve_test_path(args.test_path)
            else:
                args.test_path = 'test_data_final/global_test_data.pt'
            args.log_base  = './training_log'
            args.checkpoint_dir = './checkpoints'

        if args.data_root:
            args.data_root = resolve_data_root(args.data_root)
        if args.test_path:
            args.test_path = resolve_test_path(args.test_path)

        args.model_type = 'cnn'
        print("[INFO] Sử dụng mô hình CNN cho dữ liệu Tabular.")
        feature_extractor = CNN_FeatureExtractor(in_dim=33)
    else:
        feature_extractor = resnet18_cbam()
        args.log_base = './training_log'

    num_clients = args.num_clients
    old_client_0 = []
    old_client_1 = [i for i in range(args.num_clients)]
    new_client = []
    models = []

    ## seed settings
    setup_seed(args.seed)

    ## model settings
    model_g = network(args.numclass, feature_extractor)
    # KHÔNG đẩy model_g lên device ở đây nếu dùng đa tiến trình trên GPU ở Linux
    # model_g = model_to_device(model_g, False, args.device)
    model_old = None

    train_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ColorJitter(brightness=0.24705882352941178),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    test_transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), 
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    if args.dataset == 'cifar100':
        train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
        test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)

    elif args.dataset == 'tiny_imagenet':
        train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform, test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = train_dataset

    elif args.dataset == 'tabular':
        test_dataset = FederatedTabularDataset(client_id=0, root_dir=args.data_root, test_file=args.test_path, test=True)
        test_dataset.getTestData([0, args.numclass])
    else:
        train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = train_dataset

    if args.dataset == 'tabular':
        encode_model = CNN_Encoder(in_dim=33, num_classes=args.numclass)
        encode_model.apply(weights_init)
    else:
        encode_model = LeNet(num_classes=100)
        encode_model.apply(weights_init)

    for i in range(args.num_clients):
        if args.dataset == 'tabular':
            train_dataset = FederatedTabularDataset(client_id=i, root_dir=args.data_root, test_file=args.test_path)
        model_temp = GLFC_model(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                    args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model, i,
                    args.previous_task_replay_percent, args.seed)
        models.append(model_temp)

    ## the proxy server
    # Proxy server cũng khởi tạo trên CPU trước
    proxy_server = proxyServer(-1, args.learning_rate, args.numclass, feature_extractor, encode_model, train_transform, args.dataset)

    ## training log
    output_dir = osp.join(args.log_base, args.method, 'seed' + str(args.seed))
    os.makedirs(output_dir, exist_ok=True)

    # Tìm số thứ tự tiếp theo cho file log (log_1.txt, log_2.txt, ...)
    i = 1
    while True:
        log_filename = f'log_{i}.txt'
        log_path = osp.join(output_dir, log_filename)
        if not osp.exists(log_path):
            break
        i += 1

    out_file = open(log_path, 'w')
    print(f"\n[INFO] Bắt đầu phiên huấn luyện mới.")
    print(f"[INFO] File log được lưu tại: {log_path}\n")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_str = 'method_{}, task_size_{}, learning_rate_{}, started_at_{}'.format(
        args.method, args.task_size, args.learning_rate, timestamp)
    out_file.write(log_str + '\n')
    out_file.flush()

    classes_learned = args.task_size
    start_round = 0
    old_task_id = -1

    # Logic Resume Checkpoint
    if args.resume_path:
        if os.path.exists(args.resume_path):
            try:
                print(f"[INFO] Đang nạp checkpoint từ: {args.resume_path}")
                checkpoint = torch.load(args.resume_path, map_location='cpu', weights_only=False)

                # Cập nhật thông số học tập
                classes_learned = checkpoint['classes_learned']
                start_round = checkpoint['round'] + 1
                old_task_id = checkpoint['task_id']

                if 'old_client_0' in checkpoint:
                    old_client_0 = checkpoint['old_client_0']
                    old_client_1 = checkpoint['old_client_1']
                    new_client = checkpoint['new_client']

                # Khởi tạo lại cấu trúc mô hình tăng trưởng nếu cần
                print(f"[DEBUG] Khởi tạo lại model_g với {classes_learned} lớp")
                model_g.Incremental_learning(classes_learned)
                model_g.load_state_dict(checkpoint['model_state_dict'])

                print(f"[DEBUG] Khởi tạo lại encode_model với {classes_learned} lớp")
                encode_model.Incremental_learning(classes_learned)

                proxy_server.numclass = classes_learned
                print(f"[DEBUG] Khởi tạo lại proxy_server.model với {classes_learned} lớp")
                proxy_server.model.Incremental_learning(classes_learned)

                # Phục hồi trạng thái Proxy Server
                if checkpoint.get('proxy_best_model_1'):
                    saved_size = checkpoint['proxy_best_model_1']['fc.weight'].shape[0]
                    print(f"[DEBUG] Đang nạp proxy_best_model_1 (size thực tế trong ckpt: {saved_size})")
                    proxy_server.best_model_1 = copy.deepcopy(model_g)
                    proxy_server.best_model_1.Incremental_learning(saved_size)
                    proxy_server.best_model_1.load_state_dict(checkpoint['proxy_best_model_1'])

                if checkpoint.get('proxy_best_model_2'):
                    saved_size = checkpoint['proxy_best_model_2']['fc.weight'].shape[0]
                    print(f"[DEBUG] Đang nạp proxy_best_model_2 (size thực tế trong ckpt: {saved_size})")
                    proxy_server.best_model_2 = copy.deepcopy(model_g)
                    proxy_server.best_model_2.Incremental_learning(saved_size)
                    proxy_server.best_model_2.load_state_dict(checkpoint['proxy_best_model_2'])

                # Phục hồi trạng thái của Clients (quan trọng để không bị quên dữ liệu cũ)
                if 'client_states' in checkpoint:
                    c_states = checkpoint['client_states']
                    for idx in range(len(models), len(c_states)):
                        temp_dataset = train_dataset
                        new_model = GLFC_model(classes_learned, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                                    args.epochs_local, args.learning_rate, temp_dataset, args.device, encode_model, idx,
                                    args.previous_task_replay_percent, args.seed)
                        models.append(new_model)

                    for idx, c_state in enumerate(c_states):
                        models[idx].exemplar_set = c_state['exemplar_set']
                        models[idx].learned_classes = c_state['learned_classes']
                        models[idx].learned_numclass = c_state['learned_numclass']

                print(f"[INFO] Đã nạp thành công. Tiếp tục từ Round {start_round}, Task {old_task_id}")

                if args.test_only:
                    print("[INFO] Chế độ Test-only. Đang tiến hành đánh giá...")
                    eval_device = f"cuda:0" if torch.cuda.is_available() else "cpu"
                    acc, metrics, loss = model_global_eval(model_g, test_dataset, old_task_id, args.task_size, eval_device)

                    train_loss_val = checkpoint.get('train_loss', 0.0)
                    res_str = (
                        'Task: {}, Round: {} | '
                        'TrainLoss: {:.4f} | EvalLoss: {:.4f} | '
                        'Acc: {:.2f}% | '
                        'Macro-F1: {:.2f}% | Weighted-F1: {:.2f}% | Micro-F1: {:.2f}%\n'
                        'Macro-precision: {:.2f}% | Weighted-precision: {:.2f}% | Micro-precision: {:.2f}%\n'
                        'Macro-recall: {:.2f}% | Weighted-recall: {:.2f}% | Micro-recall: {:.2f}%'
                    ).format(old_task_id, start_round - 1, train_loss_val, loss,
                            float(acc),
                            metrics['macro']['f1'], metrics['weighted']['f1'], metrics['micro']['f1'],
                            metrics['macro']['prec'], metrics['weighted']['prec'], metrics['micro']['prec'],
                            metrics['macro']['rec'], metrics['weighted']['rec'], metrics['micro']['rec'])
                    print(res_str)
                    return
            except Exception as exc:
                print(f"[WARN] Không thể nạp checkpoint: {exc}")
                print("[WARN] Bắt đầu huấn luyện từ đầu.")
        else:
            print(f"[WARN] Không tìm thấy checkpoint tại: {args.resume_path}")
            print("[WARN] Bắt đầu huấn luyện từ đầu.")
            if args.test_only:
                return

    for ep_g in range(start_round, args.epochs_global):
        pool_grad = []
        is_task_change = (ep_g % args.tasks_global == 0)
        model_old = proxy_server.model_back()
        task_id = ep_g // args.tasks_global

        if task_id != old_task_id and old_task_id != -1:
            if args.dataset != 'tabular':
                overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
                new_client = [i for i in range(overall_client, overall_client + args.task_size)]
                old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))
                old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
                num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
                
                # Khởi tạo mô hình cho client mới (Base hụt bước này khiến Tabular Index Error)
                for idx in range(len(models), num_clients):
                    temp_dataset = train_dataset
                    new_model = GLFC_model(classes_learned, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                                args.epochs_local, args.learning_rate, temp_dataset, args.device, encode_model, idx,
                                args.previous_task_replay_percent, args.seed)
                    models.append(new_model)
            print(old_client_0)

        if task_id != old_task_id and old_task_id != -1:
            classes_learned += args.task_size
            model_g.Incremental_learning(classes_learned)
            encode_model.Incremental_learning(classes_learned)
            model_g = model_to_device(model_g, False, args.device)
            encode_model = model_to_device(encode_model, False, args.device)
            proxy_server.numclass = classes_learned
        
        print('federated global round: {}, task_id: {}'.format(ep_g, task_id))

        w_local = []
        if args.dataset == 'tabular':
            clients_index = list(range(num_clients))
        else:
            clients_index = random.sample(range(num_clients), args.local_clients)
        print('select part of clients to conduct local training')
        print(clients_index)

        # Xử lý đa GPU (Multi-GPU setup)
        if args.device == 'auto':
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                print(f"[INFO] Đã phát hiện {num_gpus} GPU. Sẽ phân phối các client trên toàn bộ tài nguyên.")
            else:
                print("[INFO] Không tìm thấy GPU. Chạy trên CPU.")
        elif args.device == '-1':
            num_gpus = 0
        else:
            # Nếu người dùng truyền chuỗi số như "0" hoặc "0,1"
            device_ids = [int(x) for x in str(args.device).split(',')]
            num_gpus = len(device_ids)

        train_losses = []
        # Chạy huấn luyện song song cho các client được chọn
        # Giới hạn max_workers để tránh lỗi CUDA Out of Memory
        # Kaggle T4 x2 nên dùng khoảng 2-4 workers để mỗi GPU gánh 1-2 client cùng lúc là an toàn nhất
        max_workers = 2 
        
        # Đảm bảo model_g đang ở CPU trước khi fork/spawn
        model_g = model_g.cpu()
        model_g_state = model_g.state_dict()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map index -> GPU ID
            futures = []
            for i_select, idx in enumerate(clients_index):
                client_obj = models[idx]
                # Đồng bộ kiến trúc mô hình với Global cho tất cả Client được chọn
                if hasattr(client_obj, 'Incremental_learning'):
                    if client_obj.model.fc.out_features != model_g.fc.out_features:
                        client_obj.Incremental_learning(model_g.fc.out_features)

                if num_gpus > 0:
                    # Phân phối đều client vào các GPU khả dụng
                    if args.device == 'auto':
                        gpu_id = i_select % num_gpus
                    else:
                        gpu_id = device_ids[i_select % num_gpus]
                else:
                    gpu_id = -1
                
                futures.append(executor.submit(local_train_step, models[idx], idx, model_g_state, task_id, model_old, ep_g, idx in old_client_0, gpu_id, is_task_change))

            # Chờ và thu thập kết quả trả về
            for future in as_completed(futures):
                res = future.result()
                c_idx = res['index']
                
                # Cập nhật kết quả vào danh sách tổng
                w_local.append(res['state_dict'])
                train_losses.append(res['train_loss'])
                
                # Cập nhật Grad pool để reconstruction ở Server
                if res['proto_grad'] is not None:
                    for pg in res['proto_grad']:
                        pool_grad.append(pg)
                
                # QUAN TRỌNG: Cập nhật lại trạng thái đồng bộ (Exemplars, Classes) cho đối tượng Client cha
                models[c_idx].exemplar_set = res['exemplar_set']
                models[c_idx].learned_classes = res['learned_classes']
                models[c_idx].learned_numclass = res['learned_numclass']
                models[c_idx].has_data = res['has_data']
                
                if not res['has_data']:
                    print(f"   [INFO] Client {c_idx} skipping local training (no data for Task {task_id+1})")
                else:
                    print(f"   [DONE] Client {c_idx} finished local training.")

        avg_train_loss = sum(train_losses) / max(len(train_losses), 1)

        ## every participant save their current training data as exemplar set
        print('every participant start updating their exemplar set and old model...')
        participant_exemplar_storing(models, num_clients, model_g, old_client_0, task_id, clients_index)
        print('updating finishes')

        print('federated aggregation...')
        w_g_new = FedAvg(w_local)
        w_g_last = copy.deepcopy(model_g.state_dict())
        
        model_g.load_state_dict(w_g_new)

        proxy_server.model = copy.deepcopy(model_g)
        proxy_server.dataloader(pool_grad)

        if args.dataset == 'tabular':
            # Eval trên classes đã học đến task hiện tại (không phải toàn bộ 34)
            eval_device = f"cuda:0" if num_gpus > 0 else "cpu"
            acc_global, metrics, eval_loss = model_global_eval(
                model_g, test_dataset, task_id, args.task_size, eval_device)
        else:
            eval_device = f"cuda:0" if num_gpus > 0 else "cpu"
            acc_global, metrics, eval_loss = model_global_eval(
                model_g, test_dataset, task_id, args.task_size, eval_device)

        log_str = (
            'Task: {}, Round: {} | '
            'TrainLoss: {:.4f} | EvalLoss: {:.4f} | '
            'Acc: {:.2f}% | '
            'Macro-F1: {:.2f}% | Weighted-F1: {:.2f}% | Micro-F1: {:.2f}%\n'
            'Macro-precision: {:.2f}% | Weighted-precision: {:.2f}% | Micro-precision: {:.2f}%\n'
            'Macro-recall: {:.2f}% | Weighted-recall: {:.2f}% | Micro-recall: {:.2f}%'
        ).format(task_id, ep_g, avg_train_loss, eval_loss,
                float(acc_global), 
                metrics['macro']['f1'], metrics['weighted']['f1'], metrics['micro']['f1'],
                metrics['macro']['prec'], metrics['weighted']['prec'], metrics['micro']['prec'],
                metrics['macro']['rec'], metrics['weighted']['rec'], metrics['micro']['rec'])
        out_file.write(log_str + '\n')
        out_file.flush()
        print(log_str)
        old_task_id = task_id

        # Lưu Checkpoint sau mỗi Round (hoặc theo interval)
        if (ep_g + 1) % args.save_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_round_{ep_g}.pt')
            # Lưu cả bản "latest" để dễ resume
            latest_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
            
            client_states = []
            for m in models:
                client_states.append({
                    'exemplar_set': m.exemplar_set,
                    'learned_classes': m.learned_classes,
                    'learned_numclass': m.learned_numclass,
                })

            state = {
                'model_state_dict': model_g.state_dict(),
                'task_id': task_id,
                'round': ep_g,
                'classes_learned': classes_learned,
                'train_loss': avg_train_loss,
                'eval_loss': eval_loss,
                'acc': float(acc_global),
                'metrics': metrics,
                'args': args,
                'client_states': client_states,
                'proxy_best_model_1': proxy_server.best_model_1.state_dict() if proxy_server.best_model_1 else None,
                'proxy_best_model_2': proxy_server.best_model_2.state_dict() if proxy_server.best_model_2 else None,
                'old_client_0': old_client_0,
                'old_client_1': old_client_1,
                'new_client': new_client
            }
            torch.save(state, checkpoint_path)
            torch.save(state, latest_path)
            print(f"   [CHECKPOINT] Đã lưu mô hình tại: {checkpoint_path}")

if __name__ == '__main__':
    freeze_support()
    main()
