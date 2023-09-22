# python version 3.7.1
# -*- coding: utf-8 -*-

def separate_users(args, dict_users):
    
    if args.dynamic:
      new_users = {}

      for _ in range(args.num_new_users):
            
            position = len(dict_users)
            data = dict_users.pop(position-1)
            new_users[position-1] = data

      return dict_users, new_users
    
    else:
      return dict_users, {}

def merge_users(dict_users, num_new_users, new_users):
    
    for _ in range(num_new_users):
          
          position = len(dict_users)
          dict_users[position] = new_users[position]

    return dict_users
