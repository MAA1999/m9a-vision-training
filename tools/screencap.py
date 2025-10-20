import os
import sys
import time
from PIL import Image
from datetime import datetime

from maa.tasker import Tasker
from maa.toolkit import Toolkit
from maa.controller import AdbController

def main():
    # 命令行参数: loop(循环) 或 single(单次), 默认single
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    # 间隔时间(秒), 默认0秒
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    
    user_path = "./"
    data_path = "./data"

    Toolkit.init_option(user_path)

    adb_devices = Toolkit.find_adb_devices()
    if not adb_devices:
        print("No ADB device found.")
        exit()

    # for demo, we just use the first device
    device = adb_devices[0]
    controller = AdbController(
        adb_path=device.adb_path,
        address=device.address,
        screencap_methods=device.screencap_methods,
        input_methods=device.input_methods,
        config=device.config,
    )
    controller.post_connection().wait()
    print(f"controller: {device}")

    os.makedirs(data_path, exist_ok=True)

    # 截图循环
    if mode == "loop":
        print(f"循环截图模式，间隔 {interval} 秒，按 Ctrl+C 停止")
        count = 0
        try:
            while True:
                screen_array = controller.post_screencap().wait().get()
                # BGR2RGB
                if len(screen_array.shape) == 3 and screen_array.shape[2] == 3:
                    rgb_array = screen_array[:, :, ::-1]
                    img = Image.fromarray(rgb_array)
                    img.save(f"{data_path}/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png")
                    count += 1
                    print(f"已截图: {count} 张")
                if interval > 0:
                    time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n已完成 {count} 张截图")
    else:
        print("单次截图模式")
        screen_array = controller.post_screencap().wait().get()
        # BGR2RGB
        if len(screen_array.shape) == 3 and screen_array.shape[2] == 3:
            rgb_array = screen_array[:, :, ::-1]
            img = Image.fromarray(rgb_array)
            img.save(f"{data_path}/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png")
            print("截图完成")
        else:
            print("Unsupported screen array format.")

if __name__ == "__main__":
    main()
