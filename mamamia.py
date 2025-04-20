import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw

# Initial Bilirubin levels over time (considered as the baseline for jaundice recovery)
initial_x1 = np.array([1, 3, 4, 6, 8, 13, 23])
initial_y1 = np.array([5.2, 8.7, 8.8, 8.5, 6.3, 4.1, 2.3])

def process_patient_report(x2, y2):
    global initial_x1, initial_y1
    
    # Normalize y2 values to match the range of y1
    y2_normalized = (y2 - np.min(y2)) / (np.max(y2) - np.min(y2)) * (np.max(initial_y1) - np.min(initial_y1)) + np.min(initial_y1)
    
    # Align y2 with x1 via interpolation
    y2_aligned = np.interp(initial_x1, x2, y2_normalized)
    
    # Learning process
    learning_rate = 0.5
    y1_learned = initial_y1 + learning_rate * (y2_aligned - initial_y1)
    
    # Compute DTW similarity
    common_x = np.linspace(1, 23, 100)
    y1_interpolated = np.interp(common_x, initial_x1, initial_y1)
    y2_interpolated = np.interp(common_x, x2, y2_normalized)
    dtw_distance = dtw.distance(y1_interpolated, y2_interpolated)
    similarity_index = 1 / (1 + dtw_distance)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(initial_x1, initial_y1, label="Avg Jaundice Recovery (Original)", color='blue', marker='o')
    plt.plot(x2, y2_normalized, label="Patient Report (Normalized)", color='orange', marker='x')
    plt.plot(initial_x1, y1_learned, label="Learned Graph", color='green', linestyle='--', marker='s')
    plt.title(f"Graph Comparison - Similarity Index: {similarity_index:.4f}")
    plt.legend()
    plt.xlabel("No. of Days")
    plt.ylabel("Bilirubin levels (mg/dl)")
    plt.grid()
    plt.show()
    
    # Ask user if they want to update the baseline graph
    question = input("Should we learn from the patient report? (yes/no): ").strip().lower()
    if question == "yes":
        initial_y1 = y1_learned  # Update baseline values
        print("Updated baseline values:")
        for i, (x, y) in enumerate(zip(initial_x1, initial_y1)):
            print(f"Day {x}: Bilirubin Level = {y:.4f}")
    else:
        print("Graph not updated.")

def main():
    while True:
        # Input patient report
        x2 = np.array(list(map(int, input("Enter patient days (comma-separated): ").split(','))))
        y2 = np.array(list(map(float, input("Enter corresponding bilirubin levels (comma-separated): ").split(','))))
        
        process_patient_report(x2, y2)
        
        # Ask if user wants to analyze another report
        cont = input("Analyze another patient report? (yes/no): ").strip().lower()
        if cont != "yes":
            print("Exiting program.")
            break

if __name__ == "__main__":
    main()
