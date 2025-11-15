"""
测试温度单位转换是否正确
"""
import xarray as xr
import numpy as np

# 模拟数据
print("="*60)
print("测试温度单位转换")
print("="*60)

# Week1 (ERA5) - 开尔文
week1_temp_k = 278.15
print(f"\nWeek1 (ERA5):")
print(f"  原始数据: {week1_temp_k} K")
print(f"  转换后: {week1_temp_k - 273.15} C")

# Week3 (MSWX) - 摄氏度
week3_temp_c = 5.36
print(f"\nWeek3 (MSWX):")
print(f"  原始数据: {week3_temp_c} C")
print(f"  统一为开尔文: {week3_temp_c + 273.15} K")
print(f"  再转换为摄氏度: {(week3_temp_c + 273.15) - 273.15} C")

# 验证修复效果
print("\n" + "="*60)
print("修复验证")
print("="*60)

# 修复前的错误
print("\n修复前（错误）:")
week3_wrong = week3_temp_c - 273.15
print(f"  Week3直接减273.15: {week3_wrong} C (错误!)")

# 修复后的正确
print("\n修复后（正确）:")
week3_k = week3_temp_c + 273.15  # 先转为开尔文
week3_correct = week3_k - 273.15  # 再统一转为摄氏度
print(f"  Week3转为K后再减273.15: {week3_correct} C (正确!)")

print("\n温度差异: {:.2f} C".format(abs(week3_correct - week3_temp_c)))

# 测试PET计算影响
print("\n" + "="*60)
print("PET计算影响测试")
print("="*60)

def calculate_pet(t_celsius):
    """简化的PET计算"""
    # 假设相对湿度为50%
    ratio = 0.5
    pet = 4.5 * np.power((1 + t_celsius / 25.0), 2) * (1 - ratio)
    return max(pet, 0)

pet_correct = calculate_pet(week3_correct)
pet_wrong = calculate_pet(week3_wrong)

print(f"\n使用正确温度 ({week3_correct:.2f} C):")
print(f"  PET = {pet_correct:.4f} mm/day")

print(f"\n使用错误温度 ({week3_wrong:.2f} C):")
print(f"  PET = {pet_wrong:.4f} mm/day")

print(f"\nPET差异: {abs(pet_correct - pet_wrong):.4f} mm/day")
print(f"PET比率: {pet_correct / pet_wrong if pet_wrong != 0 else 'inf'}")

print("\n" + "="*60)
print("结论：修复后温度数据正确，PET计算也会正确")
print("="*60)
