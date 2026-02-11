"""
日志配置模块
使用loguru配置结构化日志系统
"""
import os
import sys
from pathlib import Path
from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    app_name: str = "cad_service"
):
    """
    配置日志系统
    :param log_level: 日志级别
    :param log_dir: 日志文件目录
    :param app_name: 应用名称
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 移除默认的控制台处理器
    logger.remove()
    
    # 添加控制台输出（带颜色）
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件输出（所有日志）
    logger.add(
        log_path / f"{app_name}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # 添加错误日志文件（只记录错误）
    logger.add(
        log_path / f"{app_name}_error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="5 MB",
        retention="90 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"日志系统已初始化，日志级别: {log_level}，日志目录: {log_path.absolute()}")


def get_logger(name: str = None):
    """
    获取logger实例
    :param name: logger名称
    :return: logger实例
    """
    if name:
        return logger.bind(name=name)
    return logger


# 初始化日志系统
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(log_level=log_level)