"""Megatron EP 正式在线运行时包。

这个包的顶层只应保留主骨架：
- host：外部入口、bootstrap、hook 安装
- lifecycle：P0/P1 生命周期主线
- runtime/contracts：对外公共 API 与配置
其余按执行面、控制面、观测面分到子目录。
"""
