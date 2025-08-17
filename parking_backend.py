# encoding:utf-8
"""
停车场管理后端系统
功能：
1. 记录车牌进出信息
2. 奇数次识别=进入，偶数次识别=驶出
3. 计算停车时长
4. 生成停车记录
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class ParkingRecord:
    """单条停车记录"""
    def __init__(self, plate_number: str, entry_time: datetime, exit_time: datetime = None):
        self.plate_number = plate_number
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.duration = None
        if exit_time:
            self.duration = exit_time - entry_time

    def to_dict(self):
        return {
            'plate_number': self.plate_number,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'duration_seconds': self.duration.total_seconds() if self.duration else None,
            'duration_formatted': self.format_duration() if self.duration else None
        }

    def format_duration(self):
        """格式化停车时长"""
        if not self.duration:
            return "未完成"
        
        total_seconds = int(self.duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours}小时{minutes}分钟{seconds}秒"

class ParkingBackend:
    """停车场管理后端"""
    
    def __init__(self, data_file="parking_data.json"):
        self.data_file = data_file
        self.current_vehicles = {}  # 当前在场车辆: {车牌号: 进入次数}
        self.parking_history = []   # 完整停车历史记录
        self.recognition_count = {} # 每个车牌的识别次数: {车牌号: 次数}
        self.load_data()

    def load_data(self):
        """从文件加载数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 加载当前车辆，将字符串转换回datetime对象
                    current_vehicles_data = data.get('current_vehicles', {})
                    self.current_vehicles = {}
                    for plate, time_str in current_vehicles_data.items():
                        self.current_vehicles[plate] = datetime.fromisoformat(time_str)
                    
                    self.recognition_count = data.get('recognition_count', {})
                    
                    # 加载历史记录
                    history_data = data.get('parking_history', [])
                    self.parking_history = []
                    for record in history_data:
                        entry_time = datetime.fromisoformat(record['entry_time'])
                        exit_time = datetime.fromisoformat(record['exit_time']) if record['exit_time'] else None
                        parking_record = ParkingRecord(record['plate_number'], entry_time, exit_time)
                        self.parking_history.append(parking_record)
                        
            except Exception as e:
                print(f"加载数据失败: {e}")
                self.reset_data()

    def save_data(self):
        """保存数据到文件"""
        try:
            # 转换当前车辆的datetime对象为字符串
            current_vehicles_serializable = {}
            for plate, entry_time in self.current_vehicles.items():
                current_vehicles_serializable[plate] = entry_time.isoformat()
            
            data = {
                'current_vehicles': current_vehicles_serializable,
                'recognition_count': self.recognition_count,
                'parking_history': [record.to_dict() for record in self.parking_history]
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存数据失败: {e}")

    def reset_data(self):
        """重置所有数据"""
        self.current_vehicles = {}
        self.parking_history = []
        self.recognition_count = {}

    def process_plate_recognition(self, plate_number: str, current_time: datetime = None) -> Dict:
        """
        处理车牌识别结果
        Args:
            plate_number: 识别到的车牌号
            current_time: 当前时间，如果不提供则使用系统时间
        Returns:
            dict: 处理结果信息
        """
        if current_time is None:
            current_time = datetime.now()
        
        # 清理车牌号（去除空格等）
        plate_number = plate_number.strip()
        
        # 更新识别次数
        if plate_number not in self.recognition_count:
            self.recognition_count[plate_number] = 0
        self.recognition_count[plate_number] += 1
        
        count = self.recognition_count[plate_number]
        is_entry = count % 2 == 1  # 奇数次为进入
        
        if is_entry:
            # 车辆进入
            result = self._handle_vehicle_entry(plate_number, current_time)
        else:
            # 车辆驶出
            result = self._handle_vehicle_exit(plate_number, current_time)
        
        self.save_data()
        return result

    def _handle_vehicle_entry(self, plate_number: str, entry_time: datetime) -> Dict:
        """处理车辆进入"""
        self.current_vehicles[plate_number] = entry_time
        
        result = {
            'action': '进入',
            'plate_number': plate_number,
            'time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'recognition_count': self.recognition_count[plate_number],
            'message': f'车牌 {plate_number} 于 {entry_time.strftime("%H:%M:%S")} 进入停车场'
        }
        
        print(f"🚗 [{result['message']}]")
        return result

    def _handle_vehicle_exit(self, plate_number: str, exit_time: datetime) -> Dict:
        """处理车辆驶出"""
        if plate_number not in self.current_vehicles:
            # 车辆未记录进入就要驶出，可能是数据丢失或首次识别就是驶出
            result = {
                'action': '驶出',
                'plate_number': plate_number,
                'time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'recognition_count': self.recognition_count[plate_number],
                'error': '未找到进入记录',
                'message': f'车牌 {plate_number} 驶出，但未找到进入记录'
            }
            print(f"⚠️  [{result['message']}]")
            return result
        
        entry_time = self.current_vehicles[plate_number]
        duration = exit_time - entry_time
        
        # 创建停车记录
        parking_record = ParkingRecord(plate_number, entry_time, exit_time)
        self.parking_history.append(parking_record)
        
        # 从当前车辆列表中移除
        del self.current_vehicles[plate_number]
        
        result = {
            'action': '驶出',
            'plate_number': plate_number,
            'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': parking_record.format_duration(),
            'duration_seconds': duration.total_seconds(),
            'recognition_count': self.recognition_count[plate_number],
            'message': f'车牌 {plate_number} 于 {exit_time.strftime("%H:%M:%S")} 驶出停车场，停车时长: {parking_record.format_duration()}'
        }
        
        print(f"🚙 [{result['message']}]")
        return result

    def get_current_vehicles(self) -> List[Dict]:
        """获取当前在场车辆列表"""
        current_time = datetime.now()
        vehicles = []
        
        for plate_number, entry_time in self.current_vehicles.items():
            duration = current_time - entry_time
            vehicles.append({
                'plate_number': plate_number,
                'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'current_duration': self._format_duration_seconds(duration.total_seconds())
            })
        
        return vehicles

    def get_parking_history(self, limit: int = None) -> List[Dict]:
        """获取停车历史记录"""
        history = [record.to_dict() for record in self.parking_history]
        
        # 按时间倒序排列
        history.sort(key=lambda x: x['exit_time'] or x['entry_time'], reverse=True)
        
        if limit:
            history = history[:limit]
            
        return history

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_records = len(self.parking_history)
        current_vehicles_count = len(self.current_vehicles)
        
        if total_records > 0:
            total_duration = sum(record.duration.total_seconds() for record in self.parking_history if record.duration)
            avg_duration = total_duration / total_records if total_records > 0 else 0
        else:
            avg_duration = 0
        
        return {
            'total_completed_parkings': total_records,
            'current_vehicles_count': current_vehicles_count,
            'average_parking_duration': self._format_duration_seconds(avg_duration),
            'total_recognitions': sum(self.recognition_count.values())
        }

    def _format_duration_seconds(self, seconds: float) -> str:
        """格式化秒数为时长字符串"""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours}小时{minutes}分钟{secs}秒"

    def clear_vehicle_data(self, plate_number: str = None):
        """清空车辆数据"""
        if plate_number:
            # 清空特定车辆数据
            if plate_number in self.current_vehicles:
                del self.current_vehicles[plate_number]
            if plate_number in self.recognition_count:
                del self.recognition_count[plate_number]
        else:
            # 清空所有数据
            self.reset_data()
        
        self.save_data()

# 使用示例和测试
if __name__ == "__main__":
    # 创建后端实例
    backend = ParkingBackend("test_parking_data.json")
    
    # 模拟车牌识别
    print("=== 停车场管理系统测试 ===")
    
    # 模拟时间
    base_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    
    # 车辆1进入
    result1 = backend.process_plate_recognition("京A12345", base_time)
    print(f"结果1: {result1}")
    
    # 车辆2进入
    result2 = backend.process_plate_recognition("京B67890", base_time + timedelta(minutes=30))
    print(f"结果2: {result2}")
    
    # 车辆1驶出
    result3 = backend.process_plate_recognition("京A12345", base_time + timedelta(hours=2))
    print(f"结果3: {result3}")
    
    # 车辆2驶出
    result4 = backend.process_plate_recognition("京B67890", base_time + timedelta(hours=3))
    print(f"结果4: {result4}")
    
    # 显示统计信息
    print("\n=== 当前在场车辆 ===")
    current = backend.get_current_vehicles()
    for vehicle in current:
        print(f"车牌: {vehicle['plate_number']}, 进入时间: {vehicle['entry_time']}, 已停车: {vehicle['current_duration']}")
    
    print("\n=== 停车历史 ===")
    history = backend.get_parking_history(5)
    for record in history:
        print(f"车牌: {record['plate_number']}, 进入: {record['entry_time']}, 驶出: {record['exit_time']}, 时长: {record['duration_formatted']}")
    
    print("\n=== 统计信息 ===")
    stats = backend.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
