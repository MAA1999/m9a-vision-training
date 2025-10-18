import os
from PIL import Image
from datetime import datetime

from maa.tasker import Tasker
from maa.toolkit import Toolkit
from maa.controller import AdbController

def main():
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

    # 循环截图并保存
    while(1):
        screen_array = controller.post_screencap().wait().get()
        # BGR2RGB
        if len(screen_array.shape) == 3 and screen_array.shape[2] == 3:
            rgb_array = screen_array[:, :, ::-1]
            img = Image.fromarray(rgb_array)
            img.save(f"{data_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        else:
            print("Unsupported screen array format.")

if __name__ == "__main__":
    main()
