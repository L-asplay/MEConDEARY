from Demo.Mec_demo import demo, demo_options

if __name__ == "__main__":
    data, sel, coslist, pilist = demo(demo_options())
    skocost, skopi = coslist[0], pilist[0]
    gedcost, gedpi = coslist[1], pilist[1]
    attcost, attpi = coslist[2], pilist[2]
    dqncost, dqnpi = coslist[3], pilist[3]
 
    print("---------------------------------------------")
    print("The date of demo :")
    print("task_data", data["task_data"])
    print( "UAV_start_pos", data[ "UAV_start_pos"])    
    print("task_position", data["task_position"])    
    print("CPU_circles", data["CPU_circles"])    
    print("IoT_resource", data["IoT_resource"])    
    print("UAV_resource", data["UAV_resource"])    
    print("time_window", data["time_window"])   
    print("dependencys", data["dependencys"])    
    print("---------------------------------------------")
    print("The route of sko :", skopi+1)
    print("The cost of sko :", skocost)
    print("---------------------------------------------")
    print("The greed route of att :", gedpi)
    print("The greed cost of att :", gedcost)
    print("---------------------------------------------")
    print("The best route of att + sko :", attpi)
    print("The best cost of att + sko :", attcost)
    print("---------------------------------------------")
    print("The route of dqn + att :", dqnpi)
    print("The cost of dqn + att :", dqncost)

# nohup python -u zcong.py --problem 'mec' --graph_size 30   >./     2>&1 &
# --load_path 'outputs/mec_30/run_20250428T132621'
# python zcong.py --problem 'mec' --graph_size 30 
