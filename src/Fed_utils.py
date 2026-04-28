import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def model_to_device(model, is_old, device):
    # Nếu device là torch.device, dùng luôn
    if isinstance(device, torch.device):
        card = device
    elif device == -1 or device == "-1" or str(device).lower() == "cpu":
        card = torch.device("cpu")
    else:
        # Nếu là số (0) hoặc chuỗi số ("0") thì chuyển thành cuda:X
        if str(device).isdigit():
            card = torch.device(f"cuda:{device}")
        else:
            # Nếu là chuỗi "cuda:0", dùng luôn
            try:
                card = torch.device(device)
            except:
                card = torch.device("cpu")
    
    return model.to(card)

def participant_exemplar_storing(clients, num, model_g, old_client, task_id, clients_index):
    for index in clients_index:
        if index in old_client:
            clients[index].beforeTrain(task_id, 0)
        else:
            clients[index].beforeTrain(task_id, 1)
        clients[index].update_new_set()

# Phiên bản chạy tuần tự (như cũ) - giữ lại để dự phòng
def local_train(clients, index, model_g, task_id, model_old, ep_g, old_client, is_task_change=False):
    clients[index].model = copy.deepcopy(model_g)

    if index in old_client:
        clients[index].beforeTrain(task_id, 0)
    else:
        clients[index].beforeTrain(task_id, 1)

    # Nếu client không có dữ liệu cho task này, bỏ qua train và bảo lưu mô hình hiện tại
    if not clients[index].has_data:
        print(f"   [INFO] Client {index} skipping local training (no data for Task {task_id+1})")
        return clients[index].model.state_dict(), None, 0.0

    clients[index].update_new_set(is_task_change)
    print(clients[index].signal)
    train_loss = clients[index].train(ep_g, model_old)
    local_model = clients[index].model.state_dict()
    proto_grad = clients[index].proto_grad_sharing()

    print('*' * 60)

    return local_model, proto_grad, train_loss

# Phiên bản dành cho chạy SONG SONG
def local_train_step(client_obj, index, model_g_state, task_id, model_old, ep_g, is_old_client, device, is_task_change=False):
    """Hàm độc lập để chạy huấn luyện 1 Client trong tiến trình riêng (multiprocessing)."""
    # Khởi tạo đối tượng device chuẩn xác
    target_dev = torch.device(f"cuda:{device}" if device >= 0 else "cpu")
    
    # Gán thiết bị cho client trước khi thực hiện bất kỳ thao tác nào
    client_obj.target_device = device
    client_obj.device = target_dev
    
    # Tiếp tục các bước khác
    # 1. Chuẩn bị dữ liệu và loader trước
    group = 0 if is_old_client else 1
    client_obj.beforeTrain(task_id, group)
    
    # 2. Kiểm tra nếu không có dữ liệu cho task này thì dừng sớm
    if not client_obj.has_data:
        return {
            'index': index,
            'state_dict': client_obj.model.state_dict(),
            'proto_grad': None,
            'train_loss': 0.0,
            'has_data': False,
            'exemplar_set': client_obj.exemplar_set,
            'learned_classes': client_obj.learned_classes,
            'learned_numclass': client_obj.learned_numclass
        }

    # 3. Cập nhật model và tập dữ liệu (Exemplars/Entropy) sau khi đã có loader
    if hasattr(client_obj.model, 'Incremental_learning') and 'fc.weight' in model_g_state:
        if client_obj.model.fc.out_features != model_g_state['fc.weight'].shape[0]:
            client_obj.model.Incremental_learning(model_g_state['fc.weight'].shape[0])
            
    client_obj.model.load_state_dict(model_g_state)
    client_obj.update_new_set(is_task_change)
    
    # 4. Thực hiện huấn luyện
    train_loss = client_obj.train(ep_g, model_old, disable_pbar=True)
    
    # 5. Kết quả (Di chuyển về CPU để tránh lỗi mismatch device khi FedAvg)
    local_state = {k: v.cpu() for k, v in client_obj.model.state_dict().items()}
    proto_grad = client_obj.proto_grad_sharing()
    if proto_grad is not None:
        # Nếu có gradient, chuyển từng phần tử về CPU
        proto_grad = [[g.cpu() if g is not None else None for g in grad_list] for grad_list in proto_grad]

    # Trả về gói kết quả để Server cập nhật lại đối tượng Client cha
    return {
        'index': index,
        'state_dict': local_state,
        'proto_grad': proto_grad,
        'train_loss': train_loss,
        'has_data': True,
        'exemplar_set': client_obj.exemplar_set,
        'learned_classes': client_obj.learned_classes,
        'learned_numclass': client_obj.learned_numclass
    }

def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg

def compute_metrics(all_preds, all_labels):
    """Compute Macro, Micro, and Weighted Precision, Recall, F1."""
    all_preds  = all_preds.cpu().long()
    all_labels = all_labels.cpu().long()
    classes = torch.unique(all_labels)

    tp = torch.zeros(len(classes))
    fp = torch.zeros(len(classes))
    fn = torch.zeros(len(classes))
    support = torch.zeros(len(classes))

    for idx, c in enumerate(classes):
        tp[idx] = ((all_preds == c) & (all_labels == c)).sum().float()
        fp[idx] = ((all_preds == c) & (all_labels != c)).sum().float()
        fn[idx] = ((all_preds != c) & (all_labels == c)).sum().float()
        support[idx] = (all_labels == c).sum().float()

    # Per-class metrics
    precision_per_class = tp / (tp + fp + 1e-8)
    recall_per_class    = tp / (tp + fn + 1e-8)
    f1_per_class        = 2 * tp / (2 * tp + fp + fn + 1e-8)

    # Macro metrics (unweighted mean)
    macro_prec = precision_per_class.mean().item() * 100
    macro_rec  = recall_per_class.mean().item() * 100
    macro_f1   = f1_per_class.mean().item() * 100

    # Weighted metrics (weighted by support)
    total_support = support.sum()
    weights = support / (total_support + 1e-8)
    weighted_prec = (precision_per_class * weights).sum().item() * 100
    weighted_rec  = (recall_per_class * weights).sum().item() * 100
    weighted_f1   = (f1_per_class * weights).sum().item() * 100

    # Micro metrics (aggregate TPs, FPs, FNs)
    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_prec = (total_tp / (total_tp + total_fp + 1e-8)).item() * 100
    micro_rec  = (total_tp / (total_tp + total_fn + 1e-8)).item() * 100
    micro_f1   = (2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)).item() * 100

    metrics = {
        'macro': {'prec': macro_prec, 'rec': macro_rec, 'f1': macro_f1},
        'weighted': {'prec': weighted_prec, 'rec': weighted_rec, 'f1': weighted_f1},
        'micro': {'prec': micro_prec, 'rec': micro_rec, 'f1': micro_f1}
    }
    return metrics

def model_global_eval(model_g, test_dataset, task_id, task_size, device):
    """Evaluate global model. Returns (accuracy, precision, recall, f1, avg_loss)."""
    model_g = model_to_device(model_g, False, device)
    model_g.eval()
    test_range = [0, task_size * (task_id + 1)]
    test_dataset.getTestData(test_range)
    print(f"   [EVAL] Đang kiểm tra trên các lớp thuộc phạm vi: {test_range}")
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128)

    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    total_loss = 0.0
    num_batches = 0
    all_preds, all_labels = [], []

    for setp, (indexs, imgs, labels) in enumerate(test_loader):
        # Sử dụng .to() để linh hoạt cho cả CPU/GPU
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model_g(imgs)
            loss = criterion(outputs, labels)
        predicts = torch.max(outputs, dim=1)[1]
        correct     += (predicts.cpu() == labels.cpu()).sum()
        total       += len(labels)
        total_loss  += loss.item()
        num_batches += 1
        all_preds.append(predicts.cpu())
        all_labels.append(labels.cpu())

    accuracy = 100 * correct / total if total > 0 else torch.tensor(0.0)
    avg_loss = total_loss / max(num_batches, 1)

    all_preds  = torch.cat(all_preds)  if all_preds  else torch.tensor([])
    all_labels = torch.cat(all_labels) if all_labels else torch.tensor([])

    metrics = compute_metrics(all_preds, all_labels) if total > 0 else None

    model_g.train()
    return accuracy, metrics, avg_loss
