import launch
import pkg_resources

# 检查并安装 rembg 库
if not launch.is_installed("rembg"):
    launch.run_pip("install rembg", "要求安装 rembg 库")

# 检查 rembg 库版本
try:
    rembg_version = pkg_resources.get_distribution("rembg").version
    print(f"已安装 rembg 版本: {rembg_version}")
except pkg_resources.DistributionNotFound:
    print("警告: 找不到 rembg 库，可能安装失败")
