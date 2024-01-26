from .mhim import MHIMSystem
import torch

# 创建一个 MHIMSystem 实例
my_system = MHIMSystem(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data)

# 加载已训练的模型参数
checkpoint = torch.load("path/to/your/model.pth")
my_system.model.load_state_dict(checkpoint['model_state_dict'])
my_system.rec_model.load_state_dict(checkpoint['rec_state_dict'])
my_system.conv_model.load_state_dict(checkpoint['conv_state_dict'])
my_system.policy_model.load_state_dict(checkpoint['policy_state_dict'])

# 设置模型为评估模式
my_system.model.eval()
my_system.rec_model.eval()
my_system.conv_model.eval()
my_system.policy_model.eval()

# 进行交互测试
my_system.interact_with_model()