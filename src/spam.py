import time

side_info_array = side_info.toarray()

t = time.time()

new_A = side_info_array @ A

print(time.time() - t)

type(new_A)