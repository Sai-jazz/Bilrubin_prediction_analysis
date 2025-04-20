import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
import cv2
from skimage import measure

# Initial Bilirubin levels over time (baseline for jaundice recovery)
initial_x1 = np.array([1, 3, 4, 6, 8, 13, 23])
initial_y1 = np.array([5.2, 8.7, 8.8, 8.5, 6.3, 4.1, 2.3])

def extract_data_from_graph(image_path):
    # Load and process image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours = measure.find_contours(binary, 0.8)
    if not contours:
        raise ValueError("No graph line detected in the image.")
    
    largest_contour = max(contours, key=lambda x: len(x))
    x_img = largest_contour[:, 1]
    y_img = img.shape[0] - largest_contour[:, 0]  # Invert y-axis
    
    # Normalize to match initial data range
    x_normalized = (x_img - np.min(x_img)) / (np.max(x_img) - np.min(x_img)) * (np.max(initial_x1) - np.min(initial_x1)) + np.min(initial_x1)
    y_normalized = (y_img - np.min(y_img)) / (np.max(y_img) - np.min(y_img)) * (np.max(initial_y1) - np.min(initial_y1)) + np.min(initial_y1)
    
    # Sort and remove duplicates
    sorted_idx = np.argsort(x_normalized)
    x_sorted = x_normalized[sorted_idx]
    y_sorted = y_normalized[sorted_idx]
    unique_idx = np.unique(x_sorted, return_index=True)[1]
    
    return x_sorted[unique_idx], y_sorted[unique_idx]

def display_extracted_points(x, y):
    print("\nExtracted Points:")
    for xi, yi in zip(x, y):
        print(f"Day {xi:.1f}: Bilirubin Level = {yi:.2f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'r-', linewidth=2, label="Extracted Graph")
    plt.scatter(x, y, color='red')  # Show points as well
    plt.title("Extracted Data Points from Graph")
    plt.xlabel("No. of Days")
    plt.ylabel("Bilirubin levels (mg/dl)")
    plt.grid()
    plt.legend()
    plt.show()

def process_patient_report(x2, y2):
    global initial_x1, initial_y1
    
    # Normalize patient data
    y2_normalized = (y2 - np.min(y2)) / (np.max(y2) - np.min(y2)) * (np.max(initial_y1) - np.min(initial_y1)) + np.min(initial_y1)
    
    # Create smooth line for patient data
    patient_x_smooth = np.linspace(min(x2), max(x2), 100)
    patient_y_smooth = np.interp(patient_x_smooth, x2, y2_normalized)
    
    # Learning process
    y2_aligned = np.interp(initial_x1, x2, y2_normalized)
    learning_rate = 0.5
    y1_learned = initial_y1 + learning_rate * (y2_aligned - initial_y1)
    
    # DTW similarity calculation
    common_x = np.linspace(1, 23, 100)
    y1_interp = np.interp(common_x, initial_x1, initial_y1)
    y2_interp = np.interp(common_x, x2, y2_normalized)
    similarity_index = 1 / (1 + dtw.distance(y1_interp, y2_interp))
    
    # Plot results with smooth lines
    plt.figure(figsize=(12, 8))
    
    # Original baseline (smooth line)
    baseline_x_smooth = np.linspace(min(initial_x1), max(initial_x1), 100)
    baseline_y_smooth = np.interp(baseline_x_smooth, initial_x1, initial_y1)
    plt.plot(baseline_x_smooth, baseline_y_smooth, 'b-', linewidth=2, label="Avg Jaundice Recovery")
    plt.scatter(initial_x1, initial_y1, color='blue')  # Show original points
    
    # Patient data (smooth line)
    plt.plot(patient_x_smooth, patient_y_smooth, 'r-', linewidth=2, label="Patient Report")
    plt.scatter(x2, y2_normalized, color='red')  # Show patient points
    
    # Learned graph (smooth line)
    learned_y_smooth = np.interp(baseline_x_smooth, initial_x1, y1_learned)
    plt.plot(baseline_x_smooth, learned_y_smooth, 'g--', linewidth=2, label="Learned Graph")
    plt.scatter(initial_x1, y1_learned, color='green')  # Show learned points
    
    plt.title(f"Graph Comparison - Similarity Index: {similarity_index:.4f}")
    plt.legend()
    plt.xlabel("No. of Days")
    plt.ylabel("Bilirubin levels (mg/dl)")
    plt.grid()
    plt.show()
    
    # Update baseline if requested
    if input("Should we learn from the patient report? (yes/no): ").strip().lower() == 'yes':
        initial_y1 = y1_learned
        print("Updated baseline values:")
        for x, y in zip(initial_x1, initial_y1):
            print(f"Day {x}: Bilirubin Level = {y:.4f}")

def main():
    while True:
        image_path = input("\nEnter graph image path (or 'exit'): ").strip()
        if image_path.lower() == 'exit':
            break
        
        try:
            x2, y2 = extract_data_from_graph(image_path)
            display_extracted_points(x2, y2)
            
            if input("\nAre points correct? (yes/no): ").strip().lower() != 'yes':
                print("Please provide a clearer image.")
                continue
                
            process_patient_report(x2, y2)
            
        except Exception as e:
            print(f"Error: {e}")
        
        if input("\nAnalyze another? (yes/no): ").strip().lower() != 'yes':
            break

if __name__ == "__main__":
    main()