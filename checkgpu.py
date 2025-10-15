import tensorflow as tf

# print(tf.__version__)
# print(tf.test.gpu_device_name())
# print(tf.config.experimental.set_visible_devices)
# print('GPU:', tf.config.list_physical_devices('GPU'))
# print('CPU:', tf.config.list_physical_devices(device_type='CPU'))

# print(tf.config.list_physical_devices('GPU'))
# print(tf.test.is_gpu_available())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Create 2 virtual GPUs with 1GB memory each
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#          tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(logical_gpus)

#     print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6*1024),tf.config.LogicalDeviceConfiguration(memory_limit=6*1024)])
logical_gpus = tf.config.list_logical_devices('GPU')
print(logical_gpus)


import timeit
# def cpu_run():
#     with tf.device('/cpu:0'):
#         cpu_a = tf.random.normal([10000, 1000])
#         cpu_b = tf.random.normal([1000, 2000])
#         c = tf.matmul(cpu_a, cpu_b)
#     return c

def gpu_run():
    with tf.device('/device:GPU:1'):
        gpu_a = tf.random.normal([10000, 1000])
        gpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(gpu_a, gpu_b)
    return c

# cpu_time = timeit.timeit(cpu_run, number=100)
gpu_time = timeit.timeit(gpu_run, number=100)
# print("cpu:", cpu_time, "  gpu:", gpu_time)

print( "  gpu:", gpu_time)
