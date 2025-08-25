def load_all_data_with_ecmwf_grid():
    """加载所有数据并统一到ECMWF网格(27×47, 1.5°分辨率)"""
    print(f"加载所有数据并统一到ECMWF网格...")
    
    # 1. 先获取ECMWF网格作为参考
    print("获取ECMWF网格参考...")
    temp_file = "Tavg_2025-07-23_weekly.tif"
    temp_remote_path = f"{NAS_CONFIG['temp_base_path']}/{temp_file}"
    temp_local_file = download_from_nas(temp_remote_path)
    
    if temp_local_file is None:
        print("错误：无法获取ECMWF参考网格")
        return None, {}, {}
    
    # 获取ECMWF网格坐标
    ecmwf_ref = rxr.open_rasterio(temp_local_file)
    target_x = ecmwf_ref.x  # 经度坐标
    target_y = ecmwf_ref.y  # 纬度坐标
    print(f"ECMWF网格: {ecmwf_ref.shape}, x范围: {target_x.min().values:.1f}-{target_x.max().values:.1f}, y范围: {target_y.min().values:.1f}-{target_y.max().values:.1f}")
    os.unlink(temp_local_file)
    
    # 2. 加载和处理观测数据
    obs_nas_file = f"{NAS_CONFIG['canglong_path']}/obs_with_dewpoint_{hindcast_start_str}_to_{hindcast_end_str}.nc"
    print(f"加载观测数据: {obs_nas_file}")
    
    obs_temp_file = download_from_nas(obs_nas_file)
    
    if obs_temp_file is None:
        print("NAS上没有观测数据，从云端下载...")
        obs_data_full = download_and_process_obs_with_dewpoint()
        if obs_data_full is None:
            print("错误：无法获取观测数据")
            return None, {}, {}
    else:
        print("从NAS成功获取观测数据")
        obs_data_full = xr.open_dataset(obs_temp_file)
        os.unlink(obs_temp_file)
    
    # 将观测数据重采样到ECMWF网格
    print("将观测数据重采样到ECMWF网格...")
    obs_temp_regridded = obs_data_full['2m_temperature'].interp(
        latitude=target_y, longitude=target_x, method='linear'
    )
    obs_precip_regridded = obs_data_full['total_precipitation'].interp(
        latitude=target_y, longitude=target_x, method='linear'
    )
    obs_dewpoint_regridded = obs_data_full['2m_dewpoint_temperature'].interp(
        latitude=target_y, longitude=target_x, method='linear'
    )
    
    # 计算PET和SPEI
    obs_pet = calculate_pet_with_dewpoint(obs_temp_regridded, obs_dewpoint_regridded)
    obs_spei = calculate_spei_simple(obs_precip_regridded, obs_pet)
    
    obs_processed = {
        'temperature': obs_temp_regridded,
        'precipitation': obs_precip_regridded,
        'dewpoint': obs_dewpoint_regridded,
        'pet': obs_pet,
        'spei': obs_spei
    }
    print(f"观测数据重采样完成 -> ECMWF网格({obs_temp_regridded.shape})")
    
    # 3. 加载CAS-Canglong数据并重采样到ECMWF网格
    canglong_data = {}
    print("加载CAS-Canglong数据并重采样到ECMWF网格...")
    
    for filename, time_idx, lead_week in canglong_configs:
        remote_path = f"{NAS_CONFIG['canglong_path']}/{filename}"
        print(f"  处理CAS-Canglong Lead{lead_week}: {filename}")
        
        temp_file = download_from_nas(remote_path)
        if temp_file:
            try:
                ds = xr.open_dataset(temp_file)
                week_data = ds.isel(time=time_idx)
                
                # 单位转换
                temp_celsius = week_data['2m_temperature'] - 273.15
                precip_mm_day = week_data['total_precipitation'] * 1000 * 24
                
                # 重采样到ECMWF网格
                temp_regridded = temp_celsius.interp(
                    latitude=target_y, longitude=target_x, method='linear'
                )
                precip_regridded = precip_mm_day.interp(
                    latitude=target_y, longitude=target_x, method='linear'
                )
                
                # 计算PET和SPEI
                if '2m_dewpoint_temperature' in week_data:
                    dewpoint_celsius = week_data['2m_dewpoint_temperature'] - 273.15
                    dewpoint_regridded = dewpoint_celsius.interp(
                        latitude=target_y, longitude=target_x, method='linear'
                    )
                    pet = calculate_pet_with_dewpoint(temp_regridded, dewpoint_regridded)
                else:
                    pet = calculate_pet_simple_fallback(temp_regridded)
                
                spei = calculate_spei_simple(precip_regridded, pet)
                
                canglong_data[f'lead{lead_week}'] = {
                    'temperature': temp_regridded,
                    'precipitation': precip_regridded,
                    'pet': pet,
                    'spei': spei
                }
                print(f"    CAS-Canglong Lead{lead_week}: 重采样完成 -> {temp_regridded.shape}")
                
            except Exception as e:
                print(f"    CAS-Canglong Lead{lead_week}: 处理失败 - {e}")
            
            finally:
                os.unlink(temp_file)
        else:
            print(f"    CAS-Canglong Lead{lead_week}: 下载失败")
    
    # 4. 加载ECMWF数据（无需重采样，原本就是目标网格）
    ecmwf_data = {}
    print("加载ECMWF数据（无需重采样）...")
    
    for temp_file, precip_file, time_idx, lead_week in ecmwf_configs:
        print(f"  处理ECMWF Lead{lead_week}: {temp_file}, {precip_file}")
        
        temp_remote_path = f"{NAS_CONFIG['temp_base_path']}/{temp_file}"
        precip_remote_path = f"{NAS_CONFIG['precip_base_path']}/{precip_file}"
        
        temp_local_file = download_from_nas(temp_remote_path)
        precip_local_file = download_from_nas(precip_remote_path)
        
        temp_files_to_cleanup = []
        if temp_local_file:
            temp_files_to_cleanup.append(temp_local_file)
        if precip_local_file:
            temp_files_to_cleanup.append(precip_local_file)
        
        if temp_local_file and precip_local_file:
            try:
                temp_data = rxr.open_rasterio(temp_local_file)
                precip_data = rxr.open_rasterio(precip_local_file)
                
                if time_idx < temp_data.sizes['band']:
                    temp_celsius = temp_data.isel(band=time_idx)
                    precip_mm_day = precip_data.isel(band=time_idx)
                    
                    print(f"    ECMWF Lead{lead_week}: 原始数据形状 {temp_celsius.shape}")
                    
                    # 尝试获取露点温度
                    dewpoint_file = temp_file.replace('Tavg_', 'Tdew_')
                    dewpoint_remote_path = f"{NAS_CONFIG['dewpoint_path']}/{dewpoint_file}"
                    dewpoint_local_file = download_from_nas(dewpoint_remote_path)
                    
                    if dewpoint_local_file is None:
                        dewpoint_remote_path_alt = f"{NAS_CONFIG['temp_base_path']}/{dewpoint_file}"
                        dewpoint_local_file = download_from_nas(dewpoint_remote_path_alt)
                    
                    if dewpoint_local_file:
                        temp_files_to_cleanup.append(dewpoint_local_file)
                        try:
                            dewpoint_data = rxr.open_rasterio(dewpoint_local_file)
                            dewpoint_celsius = dewpoint_data.isel(band=time_idx)
                            pet = calculate_pet_with_dewpoint(temp_celsius, dewpoint_celsius)
                        except:
                            pet = calculate_pet_simple_fallback(temp_celsius)
                    else:
                        pet = calculate_pet_simple_fallback(temp_celsius)
                    
                    spei = calculate_spei_simple(precip_mm_day, pet)
                    
                    ecmwf_data[f'lead{lead_week}'] = {
                        'temperature': temp_celsius,
                        'precipitation': precip_mm_day,
                        'pet': pet,
                        'spei': spei
                    }
                    print(f"    ECMWF Lead{lead_week}: 完成，数据形状 {temp_celsius.shape}")
                else:
                    print(f"    ECMWF Lead{lead_week}: 时间索引{time_idx}超出范围")
            
            except Exception as e:
                print(f"    ECMWF Lead{lead_week}: 处理失败 - {e}")
            
            finally:
                for temp_file_path in temp_files_to_cleanup:
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
        else:
            print(f"    ECMWF Lead{lead_week}: 下载失败")
    
    return obs_processed, canglong_data, ecmwf_data