import torch
import os
import glob
import re
from Fed_utils import model_global_eval, setup_seed
from myNetwork import network, CNN_FeatureExtractor
from FederatedTabularDataset import FederatedTabularDataset
from option import args_parser
import copy

def get_round_number(filename):
    match = re.search(r'round_(\d+)', filename)
    return int(match.group(1)) if match else -1

def main():
    args = args_parser()
    setup_seed(args.seed)

    # Cấu hình đường dẫn cho Kaggle nếu bật flag --kaggle (hoặc tự động phát hiện)
    if args.kaggle:
        print("[INFO] Đang chạy trong môi trường Kaggle. Tự động cấu hình đường dẫn.")
        
        # Thử đường dẫn chính xác bạn vừa cung cấp
        args.data_root = '/kaggle/input/datasets/npngn123/glfc-data3/federated_data_final'
        args.test_path = '/kaggle/input/datasets/npngn123/glfc-data3/test_data_final/global_test_data.pt'
        
        # Kiểm tra nếu đường dẫn trên không tồn tại (do tên dataset trên Kaggle có thể thay đổi nhẹ)
        if not os.path.exists(args.test_path):
            print("[WARN] Không tìm thấy file test tại đường dẫn ưu tiên. Đang quét /kaggle/input...")
            found = False
            for dirpath, dirnames, filenames in os.walk('/kaggle/input'):
                if 'global_test_data.pt' in filenames:
                    args.test_path = os.path.join(dirpath, 'global_test_data.pt')
                    # Tìm folder federated_data_final tương ứng (thường cùng root)
                    parent = os.path.dirname(os.path.dirname(dirpath))
                    potential_data = os.path.join(parent, 'federated_data_final')
                    if os.path.exists(potential_data):
                        args.data_root = potential_data
                    found = True
                    break
            if not found:
                print("[ERROR] Vẫn không tìm thấy dữ liệu test. Vui lòng kiểm tra lại Dataset đã add.")
        
        # Đường dẫn checkpoint ưu tiên theo yêu cầu của bạn
        args.checkpoint_dir = '/kaggle/input/datasets/npngn123/checkpoint-glfc/checkpoint'
        if not os.path.exists(args.checkpoint_dir):
            # Thử tìm kiếm nếu đường dẫn trên không tồn tại
            for root, dirs, files in os.walk('/kaggle/input'):
                if any(f.startswith('checkpoint_round_') for f in files):
                    args.checkpoint_dir = root
                    break
        
        args.log_base = '/kaggle/working'
        print(f"[INFO] Dữ liệu: {args.data_root}")
        print(f"[INFO] File test: {args.test_path}")
        print(f"[INFO] Checkpoints: {args.checkpoint_dir}")
    else:
        args.data_root = 'federated_data_final'
        args.test_path = 'test_data_final/global_test_data.pt'
        args.log_base = args.checkpoint_dir

    # Cấu hình mặc định cho Tabular
    if args.dataset == 'tabular':
        args.task_size = 6
        feature_extractor = CNN_FeatureExtractor(in_dim=33)
        test_dataset = FederatedTabularDataset(client_id=0, root_dir=args.data_root, test_file=args.test_path, test=True)
    else:
        print("Script hiện tại tối ưu cho Tabular. Vui lòng kiểm tra lại dataset.")
        return

    checkpoint_dir = args.checkpoint_dir
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_round_*.pt"))
    checkpoint_files.sort(key=get_round_number)

    if not checkpoint_files:
        print(f"Không tìm thấy file checkpoint nào tại {checkpoint_dir}")
        return

    output_log = os.path.join(args.log_base, "re_evaluation_results.txt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(output_log, "w") as f:
        print(f"Bắt đầu đánh giá {len(checkpoint_files)} checkpoints...")
        f.write(f"Re-evaluation Log - Found {len(checkpoint_files)} checkpoints\n")
        f.write("="*80 + "\n")

        for idx, cp_path in enumerate(checkpoint_files):
            print(f"[{idx+1}/{len(checkpoint_files)}] Đang xử lý: {os.path.basename(cp_path)}")
            checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
            
            classes_learned = checkpoint['classes_learned']
            task_id = checkpoint['task_id']
            ep_g = checkpoint['round']
            train_loss_saved = checkpoint.get('train_loss', 0.0)

            # Khởi tạo model theo số lớp đã học
            model_g = network(6, feature_extractor) # Base 6 classes
            model_g.Incremental_learning(classes_learned)
            model_g.load_state_dict(checkpoint['model_state_dict'])
            
            # Đánh giá
            acc, metrics, eval_loss = model_global_eval(model_g, test_dataset, task_id, args.task_size, device)

            log_str = (
                'Task: {}, Round: {} | '
                'TrainLoss: {:.4f} | EvalLoss: {:.4f} | '
                'Acc: {:.2f}% | '
                'Macro-F1: {:.2f}% | Weighted-F1: {:.2f}% | Micro-F1: {:.2f}%\n'
                'Macro-precision: {:.2f}% | Weighted-precision: {:.2f}% | Micro-precision: {:.2f}%\n'
                'Macro-recall: {:.2f}% | Weighted-recall: {:.2f}% | Micro-recall: {:.2f}%'
            ).format(task_id, ep_g, train_loss_saved, eval_loss,
                    float(acc), 
                    metrics['macro']['f1'], metrics['weighted']['f1'], metrics['micro']['f1'],
                    metrics['macro']['prec'], metrics['weighted']['prec'], metrics['micro']['prec'],
                    metrics['macro']['rec'], metrics['weighted']['rec'], metrics['micro']['rec'])
            
            print(log_str)
            print("-" * 40)
            f.write(log_str + "\n" + "-"*40 + "\n")
            f.flush()

    print(f"\n[XONG] Kết quả đã được lưu tại: {output_log}")

if __name__ == '__main__':
    main()
