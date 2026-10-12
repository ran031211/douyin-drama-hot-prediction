# -*- coding: utf-8 -*-
"""
启动入口
=======================================================================
用法：
    python run.py                    # 默认 0.0.0.0:5000
    WEBAPP_PORT=8000 python run.py   # 换端口
    WEBAPP_ENV=production python run.py

说明：默认监听 0.0.0.0，同局域网内其他设备可直接用
      http://<本机内网IP>:<端口> 访问，便于答辩现场用手机/平板演示。
"""
from __future__ import annotations

import logging
import socket

from app import create_app
from config import Config

logger = logging.getLogger("webapp")


def _lan_ip() -> str:
    """取本机在局域网中的出口 IP（不产生真实流量）。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    app = create_app()
    print()
    print("  本机访问 : http://127.0.0.1:%d" % Config.PORT)
    print("  局域网访问: http://%s:%d" % (_lan_ip(), Config.PORT))
    print()
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
