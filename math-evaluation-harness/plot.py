import matplotlib.pyplot as plt

# 假设的数据
K_values = [1, 2, 4, 8, 16, 32, 64]
MajK_Instruct_ARC = [80.6, 80.8, 80.8, 80.6, 80.5, 80.6, 80.6]
MajK_Base_ARC = [68.8, 68.8, 68.8, 68.4, 68.4, 68.2, 67.8]
PassK_Instruct_ARC = [80.6, 80.8, 81.0, 81.0, 81.2, 81.3, 81.6]
PassK_Base_ARC = [68.8, 68.8, 69.7, 69.7, 70.3, 70.7, 71.0]

MajK_Instruct_GPQA = [28.8, 31.9, 29.0, 27.7, 27.9, 26.8, 27.0]
MajK_Base_GPQA = [23.4, 25.9, 25.0, 24.3, 23.9, 23.9, 23.2]
PassK_Instruct_GPQA = [28.8, 31.9, 32.1, 34.6, 37.5, 39.5, 42.0]
PassK_Base_GPQA = [23.4, 25.9, 27.9, 28.8, 29.0, 31.7, 32.4]

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 绘制ARC数据
ax1.plot(K_values, MajK_Instruct_ARC, 'purple', label='Maj@K-Instruct')
ax1.plot(K_values, MajK_Base_ARC, 'orange', label='Maj@K-Base')
ax1.plot(K_values, PassK_Instruct_ARC, 'green', label='Pass@K-Instruct')
ax1.plot(K_values, PassK_Base_ARC, 'blue', label='Pass@K-Base')
ax1.set_title('ARC')
ax1.set_xlabel('K: The number of candidates')
ax1.set_ylabel('Acc (%)')
ax1.legend()
ax1.grid(True)  # 添加网格
ax1.set_xscale('log')
ax1.set_xticks(K_values)  # 设置横轴刻度
ax1.set_xticklabels(K_values)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3))



# 绘制GPQA数据
ax2.plot(K_values, MajK_Instruct_GPQA, 'purple', label='Maj@K-Instruct')
ax2.plot(K_values, MajK_Base_GPQA, 'orange', label='Maj@K-Base')
ax2.plot(K_values, PassK_Instruct_GPQA, 'green', label='Pass@K-Instruct')
ax2.plot(K_values, PassK_Base_GPQA, 'blue', label='Pass@K-Base')
ax2.set_title('GPQA')
ax2.set_xlabel('K: The number of candidates')
ax2.set_ylabel('Acc (%)')
ax2.legend()
ax2.grid(True)  # 添加网格
ax2.set_xscale('log')
ax2.set_xticks(K_values)  # 设置横轴刻度
ax2.set_xticklabels(K_values) 
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3))

plt.savefig('output.png', dpi=300, bbox_inches='tight')
# 显示图形
plt.tight_layout()
plt.show()