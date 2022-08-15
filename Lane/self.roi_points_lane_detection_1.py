    self.roi_points = np.float32([
      (int(0.456*width),int(0.544*height)), # Top-left corner
      (0, height-1), # Bottom-left corner			
      (int(0.958*width),height-1), # Bottom-right corner
      (int(0.6183*width),int(0.544*height)) # Top-right corner
    ])