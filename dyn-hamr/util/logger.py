from loguru import logger


class Logger(object):
    """Compatibility wrapper for legacy Logger.log call sites."""

    @staticmethod
    def init(log_path):
        return None

    @staticmethod
    def log(write_str, to_stdout=True):
        logger.info(str(write_str))


def log_cur_stats(stats_dict, iter=None, to_stdout=True):
    loss = stats_dict.pop("total", 0)
    Logger.log(f"LOSS: {loss:.04f}", to_stdout=to_stdout)
    for k, v in stats_dict.items():
        Logger.log(f"{k}: {v:.04f}", to_stdout=to_stdout)
    if iter is not None:
        Logger.log("======= iter %d =======" % iter, to_stdout=to_stdout)
    else:
        Logger.log("========", to_stdout=to_stdout)
