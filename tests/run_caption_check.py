import importlib.machinery
import importlib.util
import os
import sys
import traceback


def main():
    try:
        here = os.path.dirname(__file__)
        path = os.path.join(here, 'test_multimodal_integration.py')
        loader = importlib.machinery.SourceFileLoader('test_mod', path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        # Ensure project root is on sys.path so test imports resolve
        proj_root = os.path.abspath(os.path.join(here, os.pardir))
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)
        loader.exec_module(module)
        module.test_all_image_points_have_captions()
        print('TEST_PASSED')
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
