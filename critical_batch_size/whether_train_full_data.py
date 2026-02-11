# alleged_global_batch_tokens=[50000, 196000, 393000, 442000, 753000, 3000000]
# seq_len = 4096
# ngpu = 8
# samples_per_step = [token // (seq_len * ngpu) for token in alleged_global_batch_tokens]
# for batch_size_x_ga in samples_per_step:
#     if batch_size_x_ga < 
# grad_accum_steps = 1
# real_global_batch_tokens = batch_size * seq_len * ngpu * grad_accum_steps
# print(f"real_global_batch_tokens is {real_global_batch_tokens}")


alleged_global_batch_tokens = [50000, 196000, 393000, 442000, 753000, 3000000, 4000000, 6000000]
seq_len = 4096
ngpu = 8
alleged_total_data_tokens = 20000000000  # 20B tokens
max_batch_size = 17  # 根据你的 H100 显存限制调整

samples_per_step = [token // (seq_len * ngpu) for token in alleged_global_batch_tokens]

print("Global Batch Size Analysis:")
print("-" * 80)

for alleged_tokens, batch_size_x_ga in zip(alleged_global_batch_tokens, samples_per_step):
    if batch_size_x_ga <= max_batch_size:
        batch_size = batch_size_x_ga
        grad_accum_steps = 1
    else:
        batch_size = max_batch_size
        grad_accum_steps = batch_size_x_ga // max_batch_size
    
    real_global_batch_tokens = batch_size * seq_len * ngpu * grad_accum_steps
    prev_steps = alleged_total_data_tokens // alleged_tokens
    steps_with_real_tokens = alleged_total_data_tokens // real_global_batch_tokens
    real_total_data_tokens = real_global_batch_tokens * ( prev_steps )  
    
    print(f"Alleged Global Batch Tokens: {alleged_tokens:>8} tokens | "
          f"batch_size: {batch_size:>2} | "
          f"grad_accum: {grad_accum_steps:>3} | "
          f"Real Global Batch Tokens: {real_global_batch_tokens:>8} tokens | "
          f"Previous Steps: {prev_steps:>8} steps | "
          f"Real Total Data Tokens: {real_total_data_tokens:>12} tokens | "
          f"Actually Needed Steps: {steps_with_real_tokens:>8} steps")