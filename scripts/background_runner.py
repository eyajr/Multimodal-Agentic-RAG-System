import threading
import subprocess
import uuid


class BackgroundRunner:
    def __init__(self):
        self.tasks = {}

    def run_async(self, cmd):
        task_id = str(uuid.uuid4())
        task = {'logs': [], 'done': False, 'returncode': None}

        def target():
            proc = subprocess.Popen([__import__('sys').executable] + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            task['proc'] = proc
            try:
                for line in proc.stdout:
                    task['logs'].append(line)
                proc.wait()
                task['returncode'] = proc.returncode
            except Exception as e:
                task['logs'].append(f"Runner error: {e}\n")
            finally:
                task['done'] = True

        th = threading.Thread(target=target, daemon=True)
        self.tasks[task_id] = task
        th.start()
        task['thread'] = th
        return task_id

    def get_logs(self, task_id):
        task = self.tasks.get(task_id)
        if not task:
            return None, True
        return ''.join(task['logs']), task['done']


runner = BackgroundRunner()
