import torch
import torch.nn.functional as F

B, T, V = 4, 10, 5000  # Example dimensions
student_logits = torch.randn(B, T, V)
teacher_logits = torch.randn(B, T, V)
T_scale = 2.0  # temperature, for instance

# Scale logits
student_log_probs = F.log_softmax(student_logits / T_scale, dim=-1)
teacher_probs = F.softmax(teacher_logits / T_scale, dim=-1)

# Compute element-wise KL divergence and sum over vocab dimension
kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T_scale ** 2) / T
print("Shape of kl:", kl.shape)  # Expected output: torch.Size([4, 10])
