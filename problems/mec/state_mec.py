import torch,math
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateMEC(NamedTuple):
    # task
    loc: torch.Tensor
    fcycles: torch.Tensor # F
    fresourse : torch.Tensor # f
    demand: torch.Tensor  # D
    tw_left: torch.Tensor # T^s
    tw_right: torch.Tensor # T^e

    # uav
    Fresourse : torch.Tensor # fUAV
    dopet : torch.Tensor # start node

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    #fixed
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    dist: torch.Tensor
    selected_: torch.Tensor 
    prec: torch.Tensor
    succ: torch.Tensor

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    cur_coord: torch.Tensor
    cur_time: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    free: torch.Tensor

    loc_energy: torch.Tensor   # 能耗
    uav_energy: torch.Tensor
    fly_energy: torch.Tensor
    # 开关电容，本地能耗比，传输速度

    #一些参数
    Rc = 4.0  # coverage / bias
    UAV_p = 3
    # UAV fly
    height = 1
    g = 0.98  
    speed = 1.0
    quantity_uav = 2
    Cd = 0.3
    A = 0.1
    air_density = 1.225
    P_fly = air_density * A * Cd * pow(speed, 3) / 2 + quantity_uav * g * speed
    P_stay = pow(speed, 3)
    # Iot device energy compute
    switched_capacitance = 3e1
    v = 4
    # transmit
    B = 1e5
    g0 = 20
    G0 = 5
    upload_P = 3
    noise_P = -9
    hm = 0
    d_2 = pow(Rc, 2) + pow(height, 2)
    upload_speed = B * math.log2(1 + g0 * G0 * upload_P / pow(noise_P, 2) / (pow(hm, 2) + d_2))

    #compute
    """
    speed 
    P_fly 
    upload_speed 
    UAV_p 
    P_stay
    """

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))
    @property
    def selected(self):
        if self.selected_.dtype == torch.uint8:
            return self.selected_
        else:
            return mask_long2bool(self.selected_, n=self.loc.size(-2))



    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            free=self.free[key],
            cur_coord=self.cur_coord[key],
            cur_time=self.cur_time[key],
            loc_energy=self.loc_energy[key],
            uav_energy=self.uav_energy[key],
            fly_energy=self.fly_energy[key],
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
  
        loc = torch.cat([input["UAV_start_pos"], input['task_position']],dim=1) #add defaut 0
        batch_size, n_node, _ = loc.size()

        return StateMEC(
            loc=loc,
            fcycles=torch.cat([torch.zeros(batch_size,1,1).to(loc.device), input['CPU_circles']],dim=1),
            fresourse=torch.cat([torch.ones(batch_size,1,1).to(loc.device), input["IoT_resource"]],dim=1),
            demand=torch.cat([torch.zeros(batch_size,1,1).to(loc.device), input['task_data']],dim=1),
            tw_left=torch.cat([torch.zeros(batch_size,1,1).to(loc.device), input['time_window'][:,:,:1]],dim=1)  ,
            tw_right=torch.cat([torch.zeros(batch_size,1,1).to(loc.device), input['time_window'][:,:,1:],],dim=1) ,
            Fresourse=input["UAV_resource"],
            dopet=input["UAV_start_pos"],

            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None], # Add steps dimension
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            selected_=(
                torch.ones(
                    batch_size, 1, n_node,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.ones(batch_size, 1, (n_node + 63 ) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            prec=torch.zeros(batch_size, n_node, 1, device=loc.device),
            succ=torch.zeros(batch_size, n_node, 1, device=loc.device),
            prev_a = torch.ones(batch_size, 1, dtype=torch.long, device=loc.device)*(0),  # 1 is to dopet
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_node,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_node + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            free=torch.ones(batch_size, n_node, 1, device=loc.device),
            cur_coord=input["UAV_start_pos"],
            cur_time = torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            loc_energy=torch.zeros(batch_size, 1, device=loc.device),
            uav_energy=torch.zeros(batch_size, 1, device=loc.device),
            fly_energy=torch.zeros(batch_size, 1, device=loc.device),
        )
    
    
    def selectasks(self, task_indices):
        if task_indices is not None :
            batch_size, n_node, _ =self.loc.size()
            selected_ = self.selected_

            if selected_.dtype == torch.uint8:
               # selected_: (batch, 1, n_node)
                mask = torch.zeros_like(selected_)
                mask = mask.scatter(-1, task_indices.unsqueeze(1), 1)
                selected_ = selected_ & (~mask)
                selected_ = selected_.to(self.selected_.dtype)  # 保证类型一致
            else:
                def mask_long_unset(mask, values, check_set=True):
                   B, K = values.size()
                   L = mask.size(-1)
                   values_exp = values[:, :, None]  # [B, K, 1]
                   rng = torch.arange(L, device=mask.device)[None, None, :]  # [1, 1, L]
                   in_bin = (values_exp >= rng * 64) & (values_exp < (rng + 1) * 64)  # [B, K, L]
                   bit_offset = values_exp % 64  # [B, K, 1]
                   bitmask = (1 << bit_offset).long()  # [B, K, 1]
                   clear_mask = (in_bin.long() * bitmask).sum(1)  # [B, L]
                   assert ((mask[:, 0, :] & clear_mask) > 0).all(), "Bit not set"
                   clear_mask = ~(in_bin.long() * bitmask).sum(1)  # [B, L]
                   result = mask.clone()
                   result[:, 0, :] = result[:, 0, :] & clear_mask
                   return result.to(self.selected_.dtype)  # 保证类型一致

                selected_ = mask_long_unset(selected_, task_indices)
            return self._replace(selected_=selected_)
        else:
            return self


    def selectasks_uav(self, task_indices):
        if task_indices is not None :
            batch_size, n_node, _ =self.loc.size()
            n_node = n_node-1
            selected_ = ( 
                torch.zeros(
                    batch_size, 1, n_node+1,
                    dtype=torch.uint8, device=self.loc.device
                )
                if self.visited_.dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_node+1 + 63) // 64, dtype=torch.int64, device=self.loc.device)  # Ceil
            )
            
            task_indices = task_indices
            if self.selected_.dtype == torch.uint8:
                for i in range(task_indices.size(1)):
                  node = task_indices[:,i:i+1]
                  selected_ = selected_.scatter(-1, node[:, :, None], 1)
            else:
                for i in range(task_indices.size(1)):
                  node = task_indices[:,i:i+1]
                  selected_ = mask_long_scatter(selected_, node)

            return self._replace(selected_=selected_)
        else : 
            return self
    

    def initdep(self, dependency):
        
        batch_size, n_node, _ =self.loc.size()
        device=self.loc.device
        pre,suc = [0]*n_node,[0]*n_node
        if len(dependency) > 1 :
            for i in range(1,len(dependency)):
                pre[dependency[i]]=dependency[i-1]
            for i in range(len(dependency)-1):
                suc[dependency[i]]=dependency[i+1]
        prec = torch.tensor(pre).unsqueeze(0).repeat(batch_size,1).unsqueeze(-1).to(device)
        succ = torch.tensor(suc).unsqueeze(0).repeat(batch_size,1).unsqueeze(-1).to(device)

        return self._replace(prec=prec,succ=succ,free=(prec<=0))
    
    
    def locexe(self):

        freemask = ((self.selected < 1).transpose(-1,-2) & (self.prec<=0)).to(dtype=torch.uint8)  
        # batch,n_node+1,1
        tw_l = self.tw_left
        tw_r = self.tw_right
        tw_r = tw_r*(1-freemask) + (tw_l+self.fcycles/self.fresourse)*freemask 
        visited = self.visited + freemask.transpose(-1,-2)
        loc_energy = ((self.switched_capacitance * pow(self.fresourse, self.v - 1) * self.fcycles)*freemask).sum(dim=1) 
        loc_energy = self.loc_energy + loc_energy
        # 此时执行完了所有没有前驱的本地任务
        
        free_indices = freemask.squeeze(-1).nonzero(as_tuple=False)
        batch_indices = free_indices[:, 0]
        point_indices = free_indices[:, 1]
        succ_indices = self.succ[batch_indices, point_indices, 0]
        free = self.free
        free[batch_indices, succ_indices,0] = 1

        #继续把从顺序解放出来的本地任务进行执行，可以确保顺序链中此时最前面的是无人机任务
        re, free, loc_energy, tw_l, tw_r, visited = self.take_loc(free, loc_energy, tw_l, tw_r, visited)
        while(re):
            re, free, loc_energy, tw_l, tw_r, visited = self.take_loc(free, loc_energy, tw_l, tw_r, visited)
        return  self._replace(tw_left=tw_l,tw_right=tw_r,visited_=visited,loc_energy=loc_energy,free=free)   


    def take_loc(self, free, loc_energy, tw_left, tw_right, visited_): # 执行现在自由的未访问本地节点
        freeloc =  ((visited_<1).transpose(-1,-2) & (self.selected<1).transpose(-1,-2) & (free>0) ).to(dtype=torch.uint8)  
        re = torch.any(freeloc == 1).item()
        if re == False :
            return re, free, loc_energy, tw_left, tw_right, visited_
        
        prec = self.prec
        succ = self.succ
        
        prec = torch.gather(tw_right, 1, prec)
        tw_left = tw_left*(1-freeloc) +  torch.maximum(prec,tw_left)*freeloc
        tw_right = tw_right*(1-freeloc) + (tw_left+self.fcycles/self.fresourse)*freeloc 
        visited_ = visited_ + freeloc.transpose(-1,-2)
        loc_energy = loc_energy + ((self.switched_capacitance * pow(self.fresourse, self.v - 1) * self.fcycles)*freeloc).sum(dim=1) 
        
        # 通过这些索引，取出相应的信息
        free_indices = freeloc.squeeze(-1).nonzero(as_tuple=False)
        succ_indices = self.succ[free_indices[:, 0], free_indices[:, 1],0]
        free[free_indices[:, 0], succ_indices] = 1

        return re, free, loc_energy, tw_left, tw_right, visited_



    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.loc[self.ids, prev_a]
        prec = self.prec[self.ids, prev_a]
        succ = self.succ[self.ids, prev_a]
        tw_left = self.tw_left 
        tw_right = self.tw_right
        
        # 计算实际执行/结束时间
        cur_left = tw_left[self.ids, prev_a]
        fly_t = (cur_coord-self.cur_coord).norm(p=2, dim=-1)/self.speed #飞行时间
        arrl_time = (self.cur_time + fly_t)[:,:,None]
        pecc_end = torch.gather(tw_right.squeeze(-1), 1, prec.squeeze(-1)).unsqueeze(-1) * (prec > 0)
        cur_left = torch.maximum(cur_left, torch.maximum(arrl_time,pecc_end))
        wait_t = (cur_left - arrl_time).squeeze(-1) # 等待时间
        mecexe_time = self.demand[self.ids, prev_a]/self.upload_speed + self.fcycles[self.ids, prev_a]/self.Fresourse
        cur_right = cur_left + mecexe_time
        cur_time = cur_right.squeeze(1)
        # 计算能耗
        mecexe_energy =  mecexe_time.squeeze(1) * self.UAV_p 
        uav_energy = self.uav_energy + mecexe_energy
        fly_energy = self.fly_energy + fly_t*self.P_fly + wait_t*self.P_stay
        #更新实际开始结束时间
        tw_left = tw_left.scatter_(1, prev_a.unsqueeze(-1).expand(-1, -1, 1), cur_left)
        tw_right = tw_right.scatter_(1, prev_a.unsqueeze(-1).expand(-1, -1, 1), cur_right)
  
        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        #解放后继
        free = self.free.scatter(1, succ, 1)
        
        loc_energy = self.loc_energy
        #把从顺序解放出来的本地任务进行执行
        re, free, loc_energy, tw_left, tw_right, visited_ = self.take_loc(free, loc_energy, tw_left, tw_right, visited_)
        while(re):
            re, free, loc_energy, tw_left, tw_right, visited_ = self.take_loc(free, loc_energy, tw_left, tw_right, visited_)

        return self._replace(prev_a=prev_a, visited_=visited_, cur_coord=cur_coord, 
                        cur_time=cur_time, i=self.i + 1, free=free, 
                        loc_energy=loc_energy, uav_energy= uav_energy, fly_energy=fly_energy,
                        tw_left=tw_left, tw_right=tw_right)
    

    def get_final_cost(self):

        assert self.all_finished()
        back_energy = ((self.dopet-self.cur_coord).norm(p=2, dim=-1)/self.speed)*self.P_fly
        #assert self.loc_energy.mean().item() <= 1e-4 and self.uav_energy.mean().item() <= 1e-4 
        assert not torch.isnan(self.loc_energy).any(), "loc with NaN"
        assert not torch.isnan(self.uav_energy).any(), "uav with NaN"
        assert not torch.isnan(self.fly_energy).any(), "fly with NaN"
        """
        print(self.loc_energy.squeeze(dim=1)[:10])
        print(self.uav_energy.squeeze(dim=1)[:10])
        print(self.fly_energy.squeeze(dim=1)[:10])
        print(back_energy.squeeze(dim=1)[:10])
        breakpoint()
        """
        return  self.loc_energy + self.uav_energy + self.fly_energy + back_energy

    def get_timewindow(self):
        return   torch.cat([self.tw_left,self.tw_right],dim=-1)

    def all_finished(self):

        mask = (self.selected[:,:,1:] < 1) | (self.visited[:,:,1:] > 0)

        return (torch.any(mask == 0).item())==False
    
    def get_finished(self):
        tw_left = self.tw_left
        tw_right = self.tw_right
        cur_time = self.cur_time
        #batch,31,1
        arrl_time = (cur_time + (self.loc-self.cur_coord).norm(p=2, dim=-1)/self.speed)[:,:,None]
        pecc_end = torch.gather(tw_right.squeeze(-1), 1, self.prec.squeeze(-1)).unsqueeze(-1) * ((self.prec > 0).to(dtype=torch.uint8))
        left = torch.maximum(tw_left, torch.maximum(arrl_time,pecc_end))
        cost_time= self.demand/self.upload_speed + self.fcycles/self.Fresourse 
        #            本地执行               已被访问              前驱没被访问                         不符合时间窗口              
        mask = ((self.selected < 1) | (self.visited > 0) | (self.free < 0).transpose(-1,-2))| (left + cost_time > tw_right).transpose(-1,-2)

        return mask.squeeze(1).all(dim=-1)[:,None]

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        
        tw_left = self.tw_left
        tw_right = self.tw_right
        cur_time = self.cur_time
        #batch,31,1
        arrl_time = (cur_time + (self.loc-self.cur_coord).norm(p=2, dim=-1)/self.speed)[:,:,None]
        pecc_end = torch.gather(tw_right.squeeze(-1), 1, self.prec.squeeze(-1)).unsqueeze(-1) * ((self.prec > 0).to(dtype=torch.uint8))
        left = torch.maximum(tw_left, torch.maximum(arrl_time,pecc_end))
        cost_time= self.demand/self.upload_speed + self.fcycles/self.Fresourse 
        #            本地执行               已被访问              前驱没被访问                         不符合时间窗口              
        mask = ((self.selected < 1) | (self.visited > 0) | (self.free < 0).transpose(-1,-2))| (left + cost_time > tw_right).transpose(-1,-2)

        mask_loc=mask[:,:,1:]

        mask_depot =  (
                (mask_loc == 0).int().sum(-1) > 0)  # [batch_size, 1]

        return torch.cat((mask_depot[:, :, None], mask_loc), -1)
        # Hacky way to return bool or uint8 depending on pytorch version #batch,1,31

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
            self.dist[
                self.ids,
                self.prev_a
            ] +
            self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions


    def show_state(self):
        print(" ")
        print("OOOOKKKK")
        print(" ")
        return
