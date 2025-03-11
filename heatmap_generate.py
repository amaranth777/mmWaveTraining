import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def load_processed_data(data_path):
    """加载预处理后的热图数据（形状：1, 17, 256, 128）"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"文件 {data_path} 不存在！")
    data = np.load(data_path)
    if data.shape != (1, 17, 256, 128):
        raise ValueError(f"数据维度不匹配！期望： (1, 17, 256, 128)，实际：{data.shape}")
    return data.squeeze(0)

def split_heatmap(heatmap_data):
    """分割热图数据为Range-Doppler和Range-Angle部分"""
    return heatmap_data[:, :, :64], heatmap_data[:, :, 64:]

def visualize_heatmaps(range_doppler, range_angle, time_idx=0, save_dir=None):
    """可视化指定时间步的热图并保存"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Range-Doppler 热图
    img0 = axes[0].imshow(range_doppler[time_idx].T, aspect='auto', cmap='jet',
                          origin='lower', extent=[0, 256, -32, 32])
    plt.colorbar(img0, ax=axes[0], label='强度 (dB)')
    axes[0].set_title(f"时间步 {time_idx} - Range-Doppler 热图")
    axes[0].set_xlabel("距离单元 (Range Bins)")
    axes[0].set_ylabel("多普勒单元 (Doppler Bins)")

    # Range-Angle 热图
    img1 = axes[1].imshow(range_angle[time_idx].T, aspect='auto', cmap='jet',
                          origin='lower', extent=[0, 256, -60, 60])
    plt.colorbar(img1, ax=axes[1], label='强度 (dB)')
    axes[1].set_title(f"时间步 {time_idx} - Range-Angle 热图")
    axes[1].set_xlabel("距离单元 (Range Bins)")
    axes[1].set_ylabel("角度单元 (Angle Bins)")

    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f"time_step_{time_idx:02d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_animations(image_dir, gif_duration):
    """生成GIF"""
    try:
        import imageio
        # 生成GIF
        filenames = [os.path.join(image_dir, f"time_step_{i:02d}.png") for i in range(17)]
        gif_path = os.path.join(image_dir, "heatmap_animation.gif")
        with (imageio.get_writer(
                gif_path, mode='I',
                duration=gif_duration*1000,
                loop=0
            ) as writer):
            for filename in filenames:
                image = imageio.v2.imread(filename)
                writer.append_data(image)
        print(f"GIF已保存至：{gif_path}")
    except ImportError:
        print("GIF生成出错，跳过GIF生成")

if __name__ == "__main__":
    base_dir = r"D:\mmWave"
    file_name = "01_54_01"
    data_path = os.path.join(base_dir, f"{file_name}.npy")
    output_dir = os.path.join(base_dir, "heatmaps", file_name)

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. 加载数据并分割
        heatmap_data = load_processed_data(data_path)
        range_doppler, range_angle = split_heatmap(heatmap_data)
        print(f"数据加载成功！Range-Doppler形状：{range_doppler.shape}, Range-Angle形状：{range_angle.shape}")

        # 2. 生成所有热图
        for time_step in range(17):
            visualize_heatmaps(range_doppler, range_angle, time_idx=time_step, save_dir=output_dir)
        print("全部热图生成完成！")

        # 3. 生成动画
        generate_animations(output_dir, gif_duration=0.5)

    except Exception as e:
        print(f"错误发生：{str(e)}")