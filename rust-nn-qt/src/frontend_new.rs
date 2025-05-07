use std::sync::{Arc, Mutex};
use eframe::egui;
use egui_plot::{Plot, PlotPoints, Line};

/// Data for tracking training progress
#[derive(Default, Clone)]
pub struct TrainingData {
    pub epoch: u32,
    pub loss: f64,
    pub accuracy: f64,
    pub losses: Vec<f64>,
    pub accuracies: Vec<f64>,
    pub training_in_progress: bool,
    pub completed: bool,
    pub should_stop: bool,
    pub show_stop_confirm: bool,
    pub dataset_path: String,
    pub available_datasets: Vec<String>,
}

impl TrainingData {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            loss: 0.0,
            accuracy: 0.0,
            training_in_progress: false,
            completed: false,
            should_stop: false,
            show_stop_confirm: false,
            losses: Vec::new(),
            accuracies: Vec::new(),
            dataset_path: "csv/pollution_dataset5k.csv".to_string(), // Default dataset
            available_datasets: vec![
                "pollution_dataset5k.csv".to_string(),
                "pollution_dataset2k.csv".to_string(),
                "pollution_dataset1k.csv".to_string(),
            ],
        }
    }

    pub fn reset(&mut self) {
        self.epoch = 0;
        self.loss = 0.0;
        self.accuracy = 0.0;
        self.training_in_progress = true;
        self.completed = false;
        self.should_stop = false;
        self.show_stop_confirm = false;
        self.losses.clear();
        self.accuracies.clear();
    }
}

/// Configuration for the neural network
#[derive(Clone)]
pub struct NetworkConfig {
    pub epochs: usize,
    pub hidden_size: usize,
    pub learning_rate: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            hidden_size: 16,
            learning_rate: 0.01,
        }
    }
}

#[derive(Clone)]
pub struct NeuralNetworkApp {
    training_data: Arc<Mutex<TrainingData>>,
    network_config: Arc<Mutex<NetworkConfig>>,
    train_callback: Option<Arc<dyn Fn() + Send + Sync + 'static>>,
}

impl Default for NeuralNetworkApp {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralNetworkApp {
    pub fn new() -> Self {
        let app = Self {
            training_data: Arc::new(Mutex::new(TrainingData::new())),
            network_config: Arc::new(Mutex::new(NetworkConfig::default())),
            train_callback: None,
        };
        
        // Scan for available datasets on startup
        app.refresh_datasets();
        
        app
    }

    pub fn update_progress(&self, epoch: usize, loss: f64, accuracy: f64) {
        let mut data = self.training_data.lock().unwrap();
        data.epoch = epoch as u32;
        data.loss = loss;
        
        // Only update accuracy if it's valid
        if accuracy > 0.0 {
            data.accuracy = accuracy;
        } else {
            // Estimate accuracy from loss
            data.accuracy = self.estimate_accuracy(loss);
        }
        
        // Store for plotting - create local copies to avoid borrowing issues
        let current_accuracy = data.accuracy;
        data.losses.push(loss);
        data.accuracies.push(current_accuracy);
    }

    pub fn training_completed(&self, accuracy: f64) {
        let mut data = self.training_data.lock().unwrap();
        data.completed = true;
        data.training_in_progress = false;
        data.should_stop = false;  // Reset flag saat pelatihan selesai
        data.accuracy = accuracy;
    }
    
    #[allow(dead_code)]
    pub fn stop_training(&self) {
        let mut data = self.training_data.lock().unwrap();
        if data.training_in_progress {
            data.should_stop = true;
            println!("Training stop requested");
        }
    }
    
    #[allow(dead_code)]
    pub fn should_stop_training(&self) -> bool {
        let data = self.training_data.lock().unwrap();
        data.should_stop
    }

    pub fn handle_train_click(&mut self, callback: impl Fn() + Send + Sync + 'static) {
        self.train_callback = Some(Arc::new(callback));
    }
    
    pub fn get_network_config(&self) -> Arc<Mutex<NetworkConfig>> {
        self.network_config.clone()
    }
    
    pub fn get_training_data(&self) -> Arc<Mutex<TrainingData>> {
        self.training_data.clone()
    }
    
    // Calculate estimated accuracy based on loss when actual accuracy is not available
    fn estimate_accuracy(&self, loss: f64) -> f64 {
        // Simple rough estimate: higher loss usually means lower accuracy
        // This is not scientific, just for visualization
        let max_loss = 0.7; // Adjust based on your data
        let accuracy = (1.0 - (loss / max_loss).min(1.0)) * 100.0;
        accuracy.max(0.0)
    }

    // Scan the csv directory for available dataset files
    pub fn scan_csv_directory(&self) -> Vec<String> {
        let mut datasets = Vec::new();
        
        if let Ok(entries) = std::fs::read_dir("csv") {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_file() {
                        if let Some(file_name) = entry.file_name().to_str() {
                            if file_name.ends_with(".csv") {
                                datasets.push(file_name.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        // If no CSV files found, provide default dataset
        if datasets.is_empty() {
            datasets.push("pollution_dataset5k.csv".to_string());
        }
        
        datasets
    }
    
    // Update the available datasets list
    pub fn refresh_datasets(&self) {
        let datasets = self.scan_csv_directory();
        let mut data = self.training_data.lock().unwrap();
        data.available_datasets = datasets;
        
        // If current dataset is not in the list, choose the first one
        if !data.available_datasets.iter().any(|ds| data.dataset_path == format!("csv/{}", ds)) {
            if let Some(first_dataset) = data.available_datasets.first() {
                data.dataset_path = format!("csv/{}", first_dataset);
            }
        }
    }
}

impl eframe::App for NeuralNetworkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Set dark mode
        ctx.set_visuals(egui::Visuals::dark());
        
        let network_config = self.network_config.clone();
        let training_data = self.training_data.clone();
        
        let mut train_click = false;
        let mut stop_click = false;
        let mut confirm_stop = false;
        let mut cancel_stop = false;
        let mut new_dataset_path = None;
        
        // Clone all the data we need upfront to avoid borrow issues
        let data_for_ui = {
            let data = training_data.lock().unwrap();
            (
                data.epoch,
                data.loss,
                data.accuracy,
                data.training_in_progress,
                data.completed,
                data.dataset_path.clone(),
                data.available_datasets.clone(),
                data.losses.clone(),
                data.accuracies.clone(),
                data.show_stop_confirm
            )
        };
        
        let (
            epoch,
            loss,
            accuracy,
            training_in_progress,
            completed,
            dataset_path,
            available_datasets,
            losses,
            accuracies,
            show_stop_confirm
        ) = data_for_ui;
        
        // Confirmation dialog
        if show_stop_confirm {
            egui::Window::new("Confirm Stop Training")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(10.0);
                        ui.heading("Are you sure you want to stop training?");
                        ui.add_space(5.0);
                        ui.label("Current progress will be saved but training will end.");
                        ui.add_space(20.0);
                        
                        ui.horizontal(|ui| {
                            if ui.button(egui::RichText::new("Yes, Stop Training")
                                .size(16.0)
                                .color(egui::Color32::WHITE))
                                .clicked() {
                                confirm_stop = true;
                            }
                            
                            ui.add_space(20.0);
                            
                            if ui.button(egui::RichText::new("Continue Training")
                                .size(16.0)
                                .color(egui::Color32::WHITE))
                                .clicked() {
                                cancel_stop = true;
                            }
                        });
                        ui.add_space(10.0);
                    });
                });
        }
        
        if confirm_stop {
            let mut data = training_data.lock().unwrap();
            data.should_stop = true;
            data.show_stop_confirm = false;
            println!("Training stop confirmed");
        }
        
        if cancel_stop {
            let mut data = training_data.lock().unwrap();
            data.show_stop_confirm = false;
        }
        
        egui::CentralPanel::default().show(ctx, |ui| {
            // Title & Header
            ui.vertical_centered(|ui| {
                ui.add_space(10.0);
                ui.heading(egui::RichText::new("Neural Network Pollution Analysis").size(24.0));
                ui.add_space(15.0);
            });
            
            // Progress information if training
            if training_in_progress || completed {
                ui.vertical_centered(|ui| {
                    let progress_text = if training_in_progress {
                        let config = network_config.lock().unwrap();
                        format!("Epoch: {}/{} | Loss: {:.4} | Accuracy: {:.2}%", 
                                epoch, config.epochs, loss, accuracy)
                    } else {
                        format!("Training completed | Final Accuracy: {:.2}%", accuracy)
                    };
                    
                    ui.colored_label(egui::Color32::from_rgb(50, 150, 200), progress_text);
                
                    // Show dataset information
                    let mut dataset_name = dataset_path.clone();
                    if let Some(file_name) = dataset_name.strip_prefix("csv/") {
                        dataset_name = file_name.to_string();
                    }
                    ui.colored_label(egui::Color32::from_rgb(180, 180, 200), 
                                     format!("Dataset: {}", dataset_name));
                    
                    // Progress bar
                    if training_in_progress {
                        let config = network_config.lock().unwrap();
                        let progress = epoch as f32 / config.epochs as f32;
                        ui.add(egui::ProgressBar::new(progress)
                            .text(format!("{}/{}", epoch, config.epochs))
                            .animate(true));
                    }
                });
                
                ui.add_space(10.0);
            }
            
            // Separator
            ui.separator();
            
            // Network Configuration section with a header
            ui.vertical(|ui| {
                ui.vertical_centered(|ui| {
                    ui.heading(egui::RichText::new("Network Configuration").size(18.0));
                });
                ui.add_space(5.0);
                
                // Only enable configuration when not training
                let is_enabled = !training_in_progress;
                ui.set_enabled(is_enabled);
                
                // 3 Configuration controls side-by-side
                ui.horizontal(|ui| {
                    // Equal spacing for each control
                    let available_width = ui.available_width();
                    let item_width = (available_width - 40.0) / 3.0;
                    
                    ui.vertical(|ui| {
                        ui.set_width(item_width);
                        ui.colored_label(egui::Color32::from_rgb(255, 255, 255), "Epochs:");
                        let mut config = network_config.lock().unwrap();
                        ui.add_sized(
                            [item_width, 30.0],
                            egui::DragValue::new(&mut config.epochs)
                                .speed(10)
                                .clamp_range(100..=5000)
                                .prefix("Epochs: ")
                        );
                    });
                    
                    ui.vertical(|ui| {
                        ui.set_width(item_width);
                        ui.colored_label(egui::Color32::from_rgb(255, 255, 255), "Hidden Layer Size:");
                        let mut config = network_config.lock().unwrap();
                        ui.add_sized(
                            [item_width, 30.0],
                            egui::DragValue::new(&mut config.hidden_size)
                                .speed(1)
                                .clamp_range(4..=128)
                                .prefix("Neurons: ")
                        );
                    });
                    
                    ui.vertical(|ui| {
                        ui.set_width(item_width);
                        ui.colored_label(egui::Color32::from_rgb(255, 255, 255), "Learning Rate:");
                        let mut config = network_config.lock().unwrap();
                        ui.add_sized(
                            [item_width, 30.0],
                            egui::DragValue::new(&mut config.learning_rate)
                                .speed(0.001)
                                .clamp_range(0.0001..=0.1)
                                .fixed_decimals(4)
                                .prefix("Rate: ")
                        );
                    });
                });
                
                // Dataset Selection
                ui.add_space(10.0);
                ui.colored_label(egui::Color32::from_rgb(255, 255, 255), "Select Dataset:");
                
                // Extract just the filename from the path
                let mut current_dataset_name = dataset_path.clone();
                if let Some(file_name) = current_dataset_name.strip_prefix("csv/") {
                    current_dataset_name = file_name.to_string();
                }
                
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_source("dataset_selector")
                        .selected_text(current_dataset_name.clone())
                        .width(200.0)
                        .show_ui(ui, |ui| {
                            for dataset in &available_datasets {
                                if ui.selectable_label(current_dataset_name == *dataset, dataset.clone()).clicked() {
                                    new_dataset_path = Some(format!("csv/{}", dataset));
                                }
                            }
                        });
                    
                    ui.add_space(10.0);
                    
                    if ui.button("Refresh Datasets").clicked() {
                        // We'll handle this outside the UI closure
                        new_dataset_path = Some("REFRESH".to_string());
                    }
                });
            });
            
            ui.add_space(10.0);
            ui.separator();
            
            // Training Charts section
            ui.vertical(|ui| {
                ui.vertical_centered(|ui| {
                    ui.heading(egui::RichText::new("Training Charts").size(18.0));
                });
                ui.add_space(5.0);
                
                // Two charts side by side
                ui.horizontal(|ui| {
                    // Loss chart
                    let plot = Plot::new("loss_plot")
                        .height(200.0)
                        .width(ui.available_width() * 0.48)
                        .view_aspect(2.0)
                        .allow_zoom(false)
                        .allow_drag(false)
                        .show_axes([true, true])
                        .legend(egui_plot::Legend::default());
                    
                    plot.show(ui, |plot_ui| {
                        if !losses.is_empty() {
                            let points: PlotPoints = losses.iter()
                                .enumerate()
                                .map(|(i, &loss)| [i as f64, loss])
                                .collect();
                            
                            plot_ui.line(Line::new(points).name("Loss").width(2.0).color(egui::Color32::RED));
                        }
                        
                        plot_ui.text(egui_plot::Text::new(
                            egui_plot::PlotPoint::new(losses.len().max(1) as f64 * 0.5, 0.01), 
                            "Loss over Epochs"
                        ).color(egui::Color32::WHITE));
                    });
                    
                    ui.add_space(10.0);
                    
                    // Accuracy chart
                    let plot = Plot::new("accuracy_plot")
                        .height(200.0)
                        .width(ui.available_width())
                        .view_aspect(2.0)
                        .allow_zoom(false)
                        .allow_drag(false)
                        .show_axes([true, true])
                        .include_y(0.0)
                        .include_y(100.0)
                        .legend(egui_plot::Legend::default());
                    
                    plot.show(ui, |plot_ui| {
                        if !accuracies.is_empty() {
                            let points: PlotPoints = accuracies.iter()
                                .enumerate()
                                .map(|(i, &acc)| [i as f64, acc])
                                .collect();
                            
                            plot_ui.line(Line::new(points).name("Accuracy").width(2.0).color(egui::Color32::BLUE));
                        }
                        
                        plot_ui.text(egui_plot::Text::new(
                            egui_plot::PlotPoint::new(accuracies.len().max(1) as f64 * 0.5, 80.0), 
                            "Accuracy (%) over Epochs"
                        ).color(egui::Color32::WHITE));
                    });
                });
            });
            
            ui.add_space(15.0);
            ui.separator();
            
            // Control buttons - centered buttons
            ui.vertical_centered(|ui| {
                ui.add_space(15.0);
                
                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    // Create button styles
                    let button_height = 40.0;
                    let button_width = 150.0;
                    
                    // Train button
                    let can_train = !training_in_progress;
                    let btn_text = if training_in_progress {
                        "Training..."
                    } else if completed {
                        "Restart Training"
                    } else {
                        "Start Training"
                    };
                    
                    if ui.add_sized(
                        [button_width, button_height],
                        egui::Button::new(
                            egui::RichText::new(btn_text)
                            .size(16.0)
                            .color(egui::Color32::WHITE)
                        )
                        .fill(if can_train { egui::Color32::from_rgb(0, 255, 8) } else { egui::Color32::DARK_GRAY })
                    ).clicked() && can_train {
                        train_click = true;
                    }
                    
                    ui.add_space(20.0);
                    
                    // Stop button (hanya aktif saat training berjalan)
                    if ui.add_sized(
                        [button_width, button_height],
                        egui::Button::new(
                            egui::RichText::new("⏹ Stop Training")
                            .size(16.0)
                            .color(egui::Color32::WHITE)
                        )
                        .fill(if training_in_progress { egui::Color32::from_rgb(255, 0, 0) } else { egui::Color32::DARK_GRAY })
                    ).clicked() && training_in_progress {
                        stop_click = true;
                    }
                });
            });
            
            // Spacer to push "Developed by" text to bottom
            let available = ui.available_height();
            if available > 30.0 {
                ui.add_space(available - 30.0);
            }
            
            // "Developed by" text at the bottom
            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new("Developed by Group 7")
                    .size(14.0)
                    .color(egui::Color32::from_rgb(150, 150, 150))
                );
            });
        });  // End of CentralPanel
        
        // Handle dataset changes 
        if let Some(path) = new_dataset_path {
            if path == "REFRESH" {
                // Just refresh the datasets list
                self.refresh_datasets();
            } else {
                // Update the selected dataset
                let mut data = training_data.lock().unwrap();
                data.dataset_path = path.clone();
                println!("Dataset changed to: {}", path);
            }
        }
        
        // Handle training button click outside of the panel to avoid borrowing issues
        if train_click {
            let mut data = self.training_data.lock().unwrap();
            if !data.training_in_progress {
                // Reset data for new training session
                if data.completed {
                    data.reset();
                }
                data.training_in_progress = true;
                data.completed = false;
                data.epoch = 0;
                
                // Trigger training callback outside of the lock
                drop(data); // Drop the lock here to avoid deadlocks
                if let Some(callback) = &self.train_callback {
                    callback();
                }
            }
        }
        
        // Handle stop button click
        if stop_click {
            let mut data = self.training_data.lock().unwrap();
            if data.training_in_progress {
                // Show confirmation dialog instead of immediately stopping
                data.show_stop_confirm = true;
            }
        }
        
        // Request continuous repainting
        ctx.request_repaint();
    }
} 