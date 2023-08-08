import subprocess


def launch_apps():
    cmd_main = [
        "uvicorn",
        "src.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--ws",
        "websockets",
        "--reload",
    ]

    cmd_endpoint = [
        "uvicorn",
        "src.endpoint:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8001",
        "--ws",
        "websockets",
        "--reload",
    ]

    subprocess.run(cmd_main, check=True)
    subprocess.run(cmd_endpoint, check=True)


if __name__ == "__main__":
    launch_apps()
