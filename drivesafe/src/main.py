import argparse
from app import DriveSafe

def main():
    parser = argparse.ArgumentParser(description='DriveSafe - Traffic Light Detection and Safety Alert System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use')
    parser.add_argument('--speed-threshold', type=float, default=30.0,
                      help='Speed threshold for warnings in km/h')
    parser.add_argument('--distance-threshold', type=float, default=50.0,
                      help='Distance threshold for warnings in meters')
    parser.add_argument('--time-to-intersection', type=float, default=3.0,
                      help='Time to intersection threshold in seconds')
    
    args = parser.parse_args()
    
    try:
        # Initialize and start DriveSafe
        app = DriveSafe(
            camera_id=args.camera,
            speed_threshold=args.speed_threshold,
            distance_threshold=args.distance_threshold,
            time_to_intersection=args.time_to_intersection
        )
        
        print("Starting DriveSafe...")
        print("Press 'q' to quit")
        
        # Start the application
        app.start()
        
    except KeyboardInterrupt:
        print("\nStopping DriveSafe...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'app' in locals():
            app.stop()

if __name__ == '__main__':
    main() 