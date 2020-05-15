from datetime import datetime
import os
import logging
from pathlib import Path
from utils import read_json, write_json
from logger import setup_logging
import pprint


class ConfigParser:

    def __init__(self, args, timestamp=True):
        args = args.parse_args()
        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        if args.resume:
            self.resume = Path(args.resume)
            self.cfg_fname = self.resume.parent / 'config.json'
            # self.cfg_fname = Path(args.config)
        else:
            msg_no_cfg = "Config file must be specified"
            assert args.config is not None, msg_no_cfg
            self.resume = None
            self.cfg_fname = Path(args.config)

        self._config = read_json(self.cfg_fname)

        if "trainer" in self.config:
            save_dir = Path(self.config['trainer']['save_dir'])
        else:
            save_dir = Path(self.config['tester']['save_dir'])

        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S") if timestamp else ""

        model_name = self.cfg_fname.parent.stem
        exper_name = f"{model_name}-{self.cfg_fname.stem}"
        self._save_dir = save_dir / 'models' / exper_name / timestamp
        self._log_dir = save_dir / 'log' / exper_name / timestamp
        self._exper_name = exper_name
        self._args = args

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        write_json(self.config, self._save_dir / 'config.json')
        self.log_path = setup_logging(self.log_dir)

    def init(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        msg = "Overwriting kwargs given in config file is not allowed"
        assert all([k not in module_args for k in kwargs]), msg
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get(self, name, default):
        return self.config.get(name, default)

    def get_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        return logger

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    def __repr__(self):
        return pprint.PrettyPrinter().pformat(self.__dict__)
