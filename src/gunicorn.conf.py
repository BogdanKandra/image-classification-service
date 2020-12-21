from multiprocessing import cpu_count
import gevent

bind = '0.0.0.0:8060'
timeout = 120
worker_class = gevent
workers = cpu_count() // 2
