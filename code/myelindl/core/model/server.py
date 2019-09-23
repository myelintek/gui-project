import uuid
import threading
import tensorflow as tf
import traceback
import multiprocessing


class KillServer(object):
    pass


class Command(object):
    def __init__(self, cmd, *args):
        self.cmd = cmd
        self.args = args


class Request(object):
    def __init__(self, username, server_id, data):
	self.username=username
        self.server_id=server_id
        self.data = data


class ExceptionWrapper(object):
    def __init__(self, exception, trace=None):
        self.type = type(exception)
        self.trace = trace
        if hasattr(exception, 'message'):
            self.message = exception.message
        else:
            self.message = str(exception)

    def to_exception(self):
        e = self.type(self.message)
        data = {'type': str(self.type), 'message':self.message,'trace':self.trace}
        e = Exception("{}".format(data))
        return e


def check_result(result):
    if isinstance(result, ExceptionWrapper):
        raise result.to_exception()
    return result


class InferenceServer(object):

    def __init__(self, sync_manager, framework, config):
        self._config = config
        self._server = None
        self.framework = framework
        self.manager = sync_manager if sync_manager else multiprocessing.Manager()
	self.request_q = self.manager.Queue()
        self.response_q = self.manager.Queue()
        self.shutdown_event = multiprocessing.Event()
	self.process = multiprocessing.Process(target=self._worker, args=[self.request_q, self.response_q, self.shutdown_event])
	self.process.start()

    def _create_server(self):
	self._server = self.framework.create_inference_server(self._config)

    def _worker(self, request_q, response_q, shutdown_event):
        try:
            self._create_server()
            while not shutdown_event.is_set():
                request = request_q.get()
                if isinstance(request, KillServer):
                    return
                response = self._server.predict_image(request)
                response_q.put(response)
	except Exception as e:
            trace = traceback.format_exc()
            response_q.put(ExceptionWrapper(e, trace))

    def predict_image(self, image):
	self.request_q.put(image)
        return check_result(self.response_q.get())
    
    def destroy(self):
        try:
            self.shutdown_event.set()
            self.request_q.put(KillServer())
            self.process.join()
        finally:
            del self.request_q
            del self.response_q
            del self.manager


class InferenceServerPool(object):
    def __init__(self, manager):
        self.inference_server_dict = {}
        self.sync_manager = manager if manager else multiprocessing.Manager()
    
    def create_inference_server(self, username, framework, config):
        if username not in self.inference_server_dict:
            self.inference_server_dict[username] = {}
	server_id = str(uuid.uuid1())[:8]
        self.inference_server_dict[username][server_id] = InferenceServer(self.sync_manager, framework, config)
        return server_id
 
    def get_inference_server(self, username, server_id):
        import json
        if username not in self.inference_server_dict:
            raise Exception('Server not exists u user:{}, id:{}'.format(username, server_id) )
        if server_id not in self.inference_server_dict[username]:
            raise Exception('Server not exists s user:{}, id:{}'.format(username, server_id) )
        
        return self.inference_server_dict[username][server_id]
        

    def get_inference_servers(self, username):
        if username not in self.inference_server_dict:
            return {}
        return self.inference_server_dict[username].keys()

    def del_inference_server(self, username, server_id):
        if username not in self.inference_server_dict:
            raise Exception('Server not exists u user:{}, id:{}'.format(username, server_id) )
        if server_id not in self.inference_server_dict[username]:
            raise Exception('Server not exists s user:{}, id:{}'.format(username, server_id) )

	try:
	    self.inference_server_dict[username][server_id].destroy()
        finally:
            del self.inference_server_dict[username][server_id]
            if not self.inference_server_dict[username]:
                del self.inference_server_dict[username]

    def destroy(self):
        for username in self.inference_server_dict.keys():
            for sid in self.inference_server_dict[username].keys():
                self.del_inference_server(username, sid)            


class PredictHandle(object):
    def __init__(self, username, server_id, input, output, lock):
        self.input_q = input
        self.output_q = output
        self.lock = lock
        self.username = username
        self.server_id = server_id

    def predict_image(self, image):
        with self.lock:
            self.input_q.put(Request(self.username, self.server_id, image))
            return check_result(self.output_q.get())


class InferenceServerPoolProxy(object):

    def __init__(self):
	self.manager = multiprocessing.Manager()
	self.input_q = self.manager.Queue()
        self.output_q = self.manager.Queue()
        self.process = multiprocessing.Process(target=self.forkserver, args=(self.manager, self.input_q, self.output_q))
        self.lock = threading.Lock()
	self.process.start()

    def forkserver(self, manager, input_q, output):
        self.server_pool = InferenceServerPool(manager)
        while True:
            try:
                request = input_q.get()
                if isinstance(request, Command):
                    result = self.handleCommand(request)
                    self.output_q.put(result)
                elif isinstance(request, Request):
                   server = self.server_pool.get_inference_server(request.username, request.server_id) 
                   result = server.predict_image(request.data)
                   self.output_q.put(result)
                elif isinstance(request, KillServer):
                    break
            except Exception as e:
                trace = traceback.format_exc()
                self.output_q.put(ExceptionWrapper(e, trace))
        self.server_pool.destroy()
                  
    def handleCommand(self, cmd):
        if not hasattr(self.server_pool, cmd.cmd):
            raise Exception('Wrong pool method {}'.format(cmd.cmd))

        func = getattr(self.server_pool, cmd.cmd)
        result = func(*cmd.args)
        return result
    
    def create_inference_server(self, username, framework, config):
        with self.lock:
            self.input_q.put(Command('create_inference_server', username, framework, config))
            return check_result(self.output_q.get())

    def get_inference_server(self, username, server_id):
        return PredictHandle(username, server_id, self.input_q, self.output_q, self.lock)

    def get_inference_servers(self, username):
        with self.lock:
	    self.input_q.put(Command('get_inference_servers', username))
            return check_result(self.output_q.get())

    def del_inference_server(self, username, server_id):
        with self.lock:
            self.input_q.put(Command('del_inference_server', username, server_id))
            return check_result(self.output_q.get())

    def destroy(self):
        try:
            self.input_q.put(KillServer())
        finally:
            del self.input_q
    	    del self.output_q
            del self.manager

    def __del__(self):
        self.destroy()

inference_server_pool = None

def init_server():
    global inference_server_pool
    if inference_server_pool is None:
        inference_server_pool = InferenceServerPoolProxy()

init_server()


