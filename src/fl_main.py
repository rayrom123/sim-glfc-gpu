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

def main():
    args = args_parser()

    ## parameters for learning
    if args.dataset == 'tabular':
        from FederatedTabularDataset import FederatedTabularDataset
        from myNetwork import MLP_FeatureExtractor, MLP_Encoder, CNN_FeatureExtractor, CNN_Encoder
        
        # Thiết lập giá trị mặc định "thông minh" cho Tabular nếu người dùng không truyền tham số khác
        if args.num_clients == 30: args.num_clients = 10
        if args.tasks_global == 10: args.tasks_global = 6
        if args.epochs_global == 100: args.epochs_global = 36
        if args.epochs_local == 20: args.epochs_local = 5
        
        args.numclass     = 34
        args.task_size    = 6
        args.learning_rate = 0.1   # LR nhỏ hơn phù hợp với MLP/tabular/CNN

        # Cấu hình đường dẫn cho Kaggle nếu bật flag --kaggle (hoặc tự động phát hiện)
        if args.kaggle:
            print("[INFO] Đang chạy trong môi trường Kaggle. Tự động cấu hình đường dẫn.")
            
            # Các gợi ý đường dẫn phổ biến trên Kaggle
            potential_roots = [
                '/kaggle/input/datasets/npngn123/glfc-data',
                '/kaggle/input/glfc-data',
                '/kaggle/input'
            ]
            
            found = False
            for root in potential_roots:
                if not os.path.exists(root): continue
                
                for dirpath, dirnames, filenames in os.walk(root):
                    if 'federated_continual_data' in dirnames:
                        args.data_root = os.path.join(dirpath, 'federated_continual_data')
                        # Thử tìm file test trong cùng thư mục hoặc thư mục cha
                        test_file = '30_test_data.pt'
                        if test_file in filenames:
                            args.test_path = os.path.join(dirpath, test_file)
                        elif test_file in os.listdir(os.path.dirname(dirpath)):
                            args.test_path = os.path.join(os.path.dirname(dirpath), test_file)
                        else:
                            # Default fallback
                            args.test_path = os.path.join(dirpath, test_file)
                            
                        args.log_base  = '/kaggle/working/training_log'
                        print(f"[INFO] Đã tìm thấy dữ liệu tại: {args.data_root}")
                        print(f"[INFO] Đã tìm thấy file test tại: {args.test_path}")
                        found = True
                        break
                if found: break
            
            if not found:
                print("[WARN] Không tìm thấy dữ liệu tự động. Sử dụng đường dẫn mặc định.")
                args.data_root = '/kaggle/input/glfc-data/federated_continual_data'
                args.test_path = '/kaggle/input/glfc-data/30_test_data.pt'
                args.log_base  = '/kaggle/working/training_log'
        else:
            args.data_root = '../federated_continual_data'
            args.test_path = '../30_test_data.pt'
            args.log_base  = './training_log'

        if args.model_type == 'cnn':
            print("[INFO] Sử dụng mô hình CNN cho dữ liệu Tabular.")
            feature_extractor = CNN_FeatureExtractor(in_dim=32)
        else:
            print("[INFO] Sử dụng mô hình MLP cho dữ liệu Tabular.")
            feature_extractor = MLP_FeatureExtractor(in_dim=32, hidden=128)
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
        if args.model_type == 'cnn':
            encode_model = CNN_Encoder(in_dim=32, num_classes=args.numclass)
        else:
            encode_model = MLP_Encoder(in_dim=32, hidden=128, num_classes=args.numclass)
        encode_model.apply(weights_init)
    else:
        encode_model = LeNet(num_classes=100)
        encode_model.apply(weights_init)

    for i in range(args.num_clients):
        if args.dataset == 'tabular':
            train_dataset = FederatedTabularDataset(client_id=i, root_dir=args.data_root, test_file=args.test_path)
        model_temp = GLFC_model(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                    args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model, i)
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
    old_task_id = -1
    for ep_g in range(args.epochs_global):
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
            print(old_client_0)

        if task_id != old_task_id and old_task_id != -1:
            if args.dataset != 'tabular':
                classes_learned += args.task_size
                model_g.Incremental_learning(classes_learned)
            model_g = model_to_device(model_g, False, args.device)
        
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
            acc_global, prec, rec, f1, eval_loss = model_global_eval(
                model_g, test_dataset, task_id, args.task_size, eval_device)
        else:
            eval_device = f"cuda:0" if num_gpus > 0 else "cpu"
            acc_global, prec, rec, f1, eval_loss = model_global_eval(
                model_g, test_dataset, task_id, args.task_size, eval_device)

        log_str = (
            'Task: {}, Round: {} | '
            'TrainLoss: {:.4f} | EvalLoss: {:.4f} | '
            'Acc: {:.2f}% | Prec: {:.2f}% | Rec: {:.2f}% | F1: {:.2f}%'
        ).format(task_id, ep_g, avg_train_loss, eval_loss,
                float(acc_global), prec, rec, f1)
        out_file.write(log_str + '\n')
        out_file.flush()
        print(log_str)
        print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, float(acc_global)))

        old_task_id = task_id

if __name__ == '__main__':
    freeze_support()
    main()
