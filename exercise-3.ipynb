#!/usr/bin/env python
# coding: utf-8

# # Exercise 3: Multiple Requests and Callbacks
# 
# In this exercise, you will implement inference with multiple requests using callbacks. 
# 
# The workload will be once again vehicle detection, but on a video this time. 
# Specifically, your application will count the cars in the frame and report three metrics: maximum number of cars in one frame, minimum number of cars in one frame, and average number of cars in all frames.
# Run the following cell to see the video.

# In[1]:


from IPython.core.display import HTML
HTML("<video alt=\"\" controls autoplay height=\"480\"><source src=\"cars_1900.mp4\" type=\"video/mp4\" /></video>")


# ### Important! The quiz will ask you the average number of vehicles detected in the last step.
# 
# 
# ## Implementation
# 
# The video course covered some potential implementations using the `wait()` function, including the zero timeout wait.
# While the zero timeout example in the video works well, it goes through all the requests over and over again until one of them is done.
# In this exercise, you will implement multiple inference that simply waits for the first finished slot using Python queues and inference callbacks
# 
# Python queues have a couple of interesting features that make them work well with the multiple request inference workload.
# One is that Python queues are thread-safe. 
# Without going in to too much detail, this means that the queue is safe to use in an asynchronous setting, like our requests.
# The second feature is the get() function (like a "pop" function). If the queue is empty when get() is called, it will wait until an item becomes available. We will begin with an optional section for those who are unfamiliar or need a review of Python Queue.
# 
# ## (Optional) Step 1: Python queue
# 
# This section is designed to give you a brief introduction to Python queue. 
# If you are already familiar, skip to step 2.
# 
# Python queues are data structures that are accessed in First In First Out (FIFO) order.
# When used in an asynchronous workload, this can be used to access the jobs as they complete.
# 
# The following is a brief example of using queue in an asynchronous setting.
# This example uses threading instead of inference engine, to keep the example simple.
# Each thread sleeps for some time, and then puts a tuple containing the id of the thread and how long it slept for.
# The main thread will wait on the queue, and print out the contents of the tuple.

# In[2]:


import queue
import threading
import time

# Sample asynchronous workload that simply sleeps, and then places ID in queue
def foo(q, myid, timeout):
    time.sleep(timeout)
    q.put((myid, timeout))

# Creating the queue for completed tasks
completion_queue = queue.Queue()

# Create and start two tasks
t1 = threading.Thread(target=foo, args=(completion_queue, 1, 3))
t2 = threading.Thread(target=foo, args=(completion_queue, 2, 1))
t1.start()
t2.start()

# Print tasks as they complete
completed_id, timeout = completion_queue.get()
print("task {} completed after sleeping for {} second(s)".format(completed_id, timeout))
completed_id, timeout = completion_queue.get()
print("task {} completed after sleeping for {} second(s)".format(completed_id, timeout))


# Confirming the threads are completed. Not necessary, but good practice.
t1.join()
t2.join()


# Notice that task 2 had a shorter timeout and completed first. It was printed immediately without waiting for task 1 to complete. Additionally, notice that I did not have to specify any ID in the `get()` function. We will adapt this for inference engine.

# ## Step 2: Inference Server mock-up
# 
# Now we will create a mock-up model of an inference server that can run multiple requests at once.
# In exercise-2 we had multiple concurrent requests by starting a number of them and then waiting for all to complete.
# However as we discussed in the video, this can be inefficient because some inferences may finish before others.
# This is especially true if you are using multiple types of devices.
# 
# So to get around this issue, we will set up this server to start inference in a request slot as soon as it is available.
# To do this, the server will keep a Python queue that has *available request slots*.
# More specifically, each item in the queue contains the ID of the available request slot.
# In addition, we will also add the status code of the inference so that the server will know if any request was unsuccessful.
# 
# The queue will be populated using the callback function for the request slot. To recap, this is the function that gets called as soon as inference is completed on the request slot. So, we will use this callback functionality to add the ID and the status (as a tuple) of the newly completed request slot.
# 
# ### utils.py
# 
# Begin by writing various helper functions for use in the main loop.
# Complete the `utils.py` file by following the instructions.
# 
# </br><details>
#     <summary><b>(2.1)</b> Complete the <code>prepImage()</code> function by finding the NCHW values from the network.</summary>
#     
# Complete the `prepImage()` function by getting the values for `n`, `c`, `h` and `w` from the function input `net`.
# The code here should be the exact same as in exercise 1.
# 
# </details><br/>
# 
# <details>
#     <summary><b>(2.2)</b> Complete the <code>createExecNetwork()</code> function which takes IECore, IENetwork and device string and returns an ExecutableNetwork with the optimal number of requests.</summary>
# 
# To get the optimal number of requests, you first need a default ExecutableNetwork object.
# The IENetwork and device string is provided as input argument.
# Use these along with IECore to get an ExecutableNetwork.
# 
# Then you can get the optimal number of requests from a metric of the ExecutableNetwork. 
# See the slides for video 2 of course 2 for more details.
# Use this value to recreate an ExecutablkeNetwork object with the optimal number of requests.
# Finally, return this executable network.
# 
# </details><br/>
# 
# <details>
#     <summary><b>(2.3)</b> In <code>setCallbackAndQueue()</code> function, add a callback function called <code>callbackFunc</code> to each of the request slots. </summary>
#     
# We will be defining `callbackFunc` function in step (2.4), but we will work on the part where this callback is added to the request slot.
# 
# For our callback function, we need two pieces of information: the ID of the request slot, and the status of the inference.
# Additionally, we need access to the queue that keeps track of the completed slots.
# The status is automatically made available to the callback function, but the request slot ID as well as access to the queue is not.
# So we need to pass these to the function.
# 
# To do this, we need to use the `py_data` variable. 
# This dictionary variable is set when you add the callback, and is passed in as an argument to the callback function.
# For what we need in our callback function, `py_data` must contain the ID of the request slot and the queue.
# 
# So first create a dictionary that contains these two. 
# The key to use for this dictionary is up to you. 
# Then call the  `set_completion_callback()` method for the requests to add the `callbackFunc` (note the lack of parethesis) along with the `py_data`. 
# </details><br/>
# 
# <details>
#     <summary><b>(2.4)</b> Complete <code>callbackFunc()</code> function, by having it add a tuple containing the request slot ID and the status code for the inference.</summary> 
# 
# Remember that `py_data` argument is the dictionary you passed in in the previous step.
# It should contain the queue and the request ID.
# The status of the inference is in the input argument `status`.
# Add the tuple (ID, status) to the queue. Note that the order there matters.
# 
# </details><br/>

# In[3]:


get_ipython().run_cell_magic('writefile', 'utils.py', 'import cv2\nfrom openvino.inference_engine import IECore, IENetwork\n\n# Prepares image for inference by reshaping and transposing.\n# inputs:\n#     orig_image - numpy array containing the original, unprocessed image\n#     ie_net     - IENetwork object \ndef prepImage(orig_image, ie_net):\n    \n    ##! (2.1) Find n, c, h, w from ie_net !##\n    \n    input_image = cv2.resize(orig_image, (w, h))\n    input_image = input_image.transpose((2, 0, 1))\n    input_image.reshape((n, c, h, w))\n\n    return input_image\n\n# Processes the result. Returns the number of detected vehices in the image.\n# inputs:\n#    detected_obects - numpy array containing the ooutput of the model\n#    prob_threashold - Required probability for "detection"\n# output:\n#    Number of vehices detected.\ndef getCount(detected_objects, prob_threshold=0.5):\n    detected_count = 0\n    for obj in detected_objects[0][0]:\n        # Draw only objects when probability more than specified threshold\n        if obj[2] > prob_threshold:\n            detected_count+=1\n    return detected_count\n\n\n# Create ExecutableNetwork with the optimal number of requests for a given device.\n# inputs:\n#    ie_core - IECore object to use\n#    ie_net  - IENetwork object to use\n#    device  - String to use for device_name argument.\n# output:\n#    ExecutabeNetwork object\ndef createExecNetwork(ie_core, ie_net, device):   \n    ##! (2.2) Create ExecutableNetwork object and find the optimal number of requests !##\n\n    ##! (2.2) Recreate ExecutableNetwork and with num_requests set to optimal number of requests !##\n    \n    ##! (2.2) return the ExecutableNetwork !##\n\n    \n# Set callback functions for the inference requests.\n# inputs:\n#    exec_net - ExecutableNetwork object to modify\n#    c_queue  - Python queue to put the slot ID in\ndef setCallbackAndQueue(exec_net, c_queue):\n    for req_slot in range(len(exec_net.requests)):\n        ##! (2.3) Create a dictionary for py_data to pass in the queue and ID !###\n\n        ##! (2.3) Set the completion callback with the arguments for each reqeust !##\n        \n        # Initializing the queue. The second item of the tuple is the status of the previous \n        #  inference. But as there is no previous inference right now, setting the status to None.\n        c_queue.put((req_slot, None))\n    \n# Callback function called on completion of the inference.\n# inputs:\n#    status  - status code for the inference.\n#    py_data - dictionary arguments passed into the function\ndef callbackFunc(status, py_data):\n    try:\n        ##! (2.4) Add a tuple (id, status) to queue here !##\n    except:\n        print("There was an issue in callback")')


# ### main.py
# 
# Now write the main loop. 
# Complete the `main.py` file by following the instructions.
# 
# *Note* Many of the variables are already placed and set to None. This is because these variables are used in other parts of the code that have been provided to you. So do not change the name of the variable, but instead replace None with code specified by the instructions.
# 
# 
# 
# </br><details>
#     <summary><b>(2.5)</b> Create an IECore object and use it to create IENetwork object with the provded model. Then get the input and output layer names. Use <code>ie_core</code> and <code>ie_net</code> as the variable names.</summary>
# 
# The paths for the model is provided. Do not change the variable name, `ie_core` and `ie_net` for ths file. The name of the input layer and output layer are stored in `inputs` and `outputs` dictionaries.
# 
# </details><br/>
# 
# <details><summary><b>(2.6)</b> Run <code>getCount()</code> on the request result to get the number of vehicles. </summary>
# 
# Use the request slot ID (`req_slot`) to get the result. Then get the number of vehicles from each inference request with `getCount()` function. remember that result of the inference itself can be accessed through the `outputs` attribute of the requests.
# 
# </details><br/>
# 
# <details>
#     <summary><b>(2.7)</b> Start asynchronous processing on the next image. </summary>
# 
# Asynchronous (non-blocking) inference is started with `start_async()`.
# 
# </details><br/>
# 
# <details>
# 
# <summary><b>(2.8)</b> Handle the remaining requests. </summary>
# 
# The main while loop ends as soon as there are no more images to process, but there will be some inference that is still running. 
# So we need to handle the remaining request.
# We first wait until all requests are completed, then we can handle the remaining results in the queue.
# 
# This part is already implemented, so you just need to get the result.
# This step should be identical to 2.6
# 
# </details><br/>

# In[4]:


get_ipython().run_cell_magic('writefile', 'main.py', 'import cv2\nimport sys\nimport queue\nfrom openvino.inference_engine import IECore, IENetwork\nfrom utils import *\n\ndevice = sys.argv[1]\n\n##! (2.5) Create IECore and IENetwork object from vehicle-detection-adas-0002 !##\nxml_path="/data/intel/vehicle-detection-adas-0002/FP16/vehicle-detection-adas-0002.xml"\nbin_path="/data/intel/vehicle-detection-adas-0002/FP16/vehicle-detection-adas-0002.bin"\nie_core = None\nie_net = None\n\n##! (2.5) Get the name of input and output layers. There is only one of each. !##\ninput_layer  = None\noutput_layer = None\n\n# Create ExecutableNetwork object using createExecNetwork in utils.py \nexec_net = createExecNetwork(ie_core, ie_net, device)\nprint("ExecutableNetwork created with {} requests.".format(len(exec_net.requests)))\n\n# Set the callback functions using setCallbackAndQueue() in utils.py \nc_queue = queue.Queue()\nsetCallbackAndQueue(exec_net, c_queue)\n\n# Stats for processing\nmax_vehicles = 0\nmin_vehicles = 999      # this is safe as the max number of detectable objects is 200\nsum_vehicles = 0\nnum_frames = 0\n# Loading the data from a video\ninput_video = "/data/reference-sample-data/object-detection-python/cars_1900.mp4"\ncap = cv2.VideoCapture(input_video)\nwhile cap.isOpened():\n    # Read the next frame\n    ret, next_frame = cap.read()\n    # Condition for the end of video\n    if not ret:\n        break\n        \n    ##! preprocess next_frame using prepImage from utils.py !##\n    input_frame = prepImage(next_frame, ie_net) \n    \n    # using get to wait for the nextslot ID. Here we are setting a timeout of 30 seconds in case \n    #  there are issues with the callback and queue never gets populated. With timeout, this function\n    #  will error out  with "Empty"\n    req_slot, status = c_queue.get(timeout=30)\n    \n    if status == 0:\n        ##! (2.6) Postprocess result from the request slot using getCount function from utils.py !##\n        num_vehicles = None\n        \n        max_vehicles = max(num_vehicles, max_vehicles)\n        min_vehicles = min(num_vehicles, min_vehicles)\n        sum_vehicles += num_vehicles\n        num_frames += 1\n        \n    # Recall that None is what we set for the first time initializeation of queue, so we catch everything else.\n    elif not status is None:\n        print("There was error in processing an image")\n\n    ##! (2.7) Start the next inference on the now open slot. !##\n    \n\n# Handle the remaining images.\n#  first we wait for all request slots to complete\nfor req in exec_net.requests:\n    req.wait()\n\n# Handle remaining results \nwhile not c_queue.empty():\n    req_slot, status = c_queue.get(timeout=30)\n    \n    if status == 0:\n        ##! (2.8) Postprocess result from the request slot using getCount function from utils.py !##\n        num_vehicles = None\n        \n        max_vehicles = max(num_vehicles, max_vehicles)\n        min_vehicles = min(num_vehicles, min_vehicles)\n        sum_vehicles += num_vehicles\n        \n    # Recall that None is what we set for the first time initializeation of queue, so we catch everything else.\n    elif not status is None:\n        print("There was error in processing an image")\n        \n# Finally, reporting results.\nprint("Maximum number of cars detected: {}".format(max_vehicles))\nprint("Minimum number of cars detected: {}".format(min_vehicles))\nprint("average number of cars detected: {:.3g}".format(sum_vehicles/num_frames))')


# ### job file
# 
# Once again, the job file is provided for you. Note the if statement where we set up for FPGA if it is in the device list. Run the following cell to create the bash script `run.sh` to be used for benchmarking.

# In[5]:


get_ipython().run_cell_magic('writefile', 'run.sh', '\nDEVICE=$1\nsource /opt/intel/openvino/bin/setupvars.sh\n\n# Check if FPGA is used \nif grep -q FPGA <<<"$DEVICE"; then\n    # Environment variables and compilation for edge compute nodes with FPGAs\n    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2\n    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh\n    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-3_PL2_FP16_MobileNet_Clamp.aocx\n    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3\nfi\n    \n# Running the object detection code\npython3 main.py $DEVICE')


# ## Run the job
# 
# Finally, let us try to run the workload. 
# Once again we've provided the same `submitToDevCloud` function.
# 
# **Note:** The toolkit is very verbose when using MYRIAD systems, so you may get a lot of additional output beyond what you are expecting. 
# 

# In[6]:


from devcloud_utils import submitToDevCloud
submitToDevCloud("run.sh", "VPU", script_args=["MYRIAD"], files=["main.py","utils.py"])


# Congratulations! You have just run multiple requests in parallel. 
# From the output, you can see that multiple requests are being ran in parallel. 
# 
# **The final average vehicles detected to the third decimal will be asked in the quiz.**
