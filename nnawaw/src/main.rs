use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use rand::thread_rng;
use csv::ReaderBuilder;
use std::error::Error;
use plotters::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;
use eframe;

mod frontend;
use frontend::NeuralNetworkApp;

// Default values moved to NetworkConfig in frontend.rs
const LOG_INTERVAL: usize = 100; // How often to log progress

fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.0))
}

fn relu_deriv(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn binary_cross_entropy(y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
    let eps = 1e-7;
    let y_pred_clipped = y_pred.mapv(|v| v.max(eps).min(1.0 - eps));
    let loss = y_true * &y_pred_clipped.mapv(|v| v.ln()) +
               (1.0 - y_true) * &y_pred_clipped.mapv(|v| (1.0 - v).ln());
    -loss.mean().unwrap()
}

fn load_data(path: &str) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
    // Check if file exists
    if !std::path::Path::new(path).exists() {
        return Err(format!("File not found: {}", path).into());
    }
    
    // Baca beberapa baris pertama untuk mendeteksi delimiter
    let file = std::fs::File::open(path)?;
    let mut buf_reader = std::io::BufReader::new(file);
    let mut first_line = String::new();
    std::io::BufRead::read_line(&mut buf_reader, &mut first_line)?;
    
    // Deteksi delimiter: jika ada titik koma, gunakan titik koma, jika tidak gunakan koma
    let delimiter = if first_line.contains(';') { b';' } else { b',' };
    println!("Detected delimiter: '{}'", char::from(delimiter));

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delimiter)
        .from_path(path)?;

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let vals: Result<Vec<f64>, _> = record.iter().map(|s| s.trim().parse::<f64>()).collect();
        if let Ok(vals) = vals {
            if !vals.is_empty() {
                let (x, y) = vals.split_at(vals.len() - 1);
                features.push(x.to_vec());
                labels.push(y[0]);
            }
        }
    }

    // Check if we have any data
    if features.is_empty() {
        return Err(format!("No valid data found in {}", path).into());
    }

    // Check if all feature vectors have the same length
    let feature_len = features[0].len();
    if features.iter().any(|f| f.len() != feature_len) {
        return Err("Inconsistent feature dimensions in dataset".into());
    }

    let feature_array = Array2::from_shape_vec((features.len(), features[0].len()), features.concat())?;
    let label_array = Array2::from_shape_vec((labels.len(), 1), labels)?;

    println!("Successfully loaded dataset from {} with {} samples and {} features", 
             path, features.len(), feature_len);

    Ok((feature_array, label_array))
}

fn plot_loss(losses: &[f64], epochs: usize) -> Result<(), Box<dyn Error>> {
    // Create result directory if it doesn't exist
    std::fs::create_dir_all("result")?;

    let root = BitMapBackend::new("result/lossfigure.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = losses.iter().cloned().fold(f64::NAN, f64::max);
    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..epochs, 0.0..max_loss)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &loss)| (i, loss)),
        &RED,
    ))?;

    Ok(())
}

fn train_neural_network(
    app: Arc<Mutex<NeuralNetworkApp>>,
) -> Result<(), Box<dyn Error>> {
    // Get configuration
    let config = {
        let app_locked = app.lock().unwrap();
        let config_ref = app_locked.get_network_config();
        let config = config_ref.lock().unwrap().clone();
        config
    };
    
    let epochs = config.epochs;
    let hidden_size = config.hidden_size;
    let learning_rate = config.learning_rate;
    
    println!("Starting training with: Epochs={}, Hidden Size={}, Learning Rate={}", 
             epochs, hidden_size, learning_rate);
    
    let (x, y_true) = load_data("csv/pollution_dataset5k.csv")?;
    let (n_samples, n_features) = x.dim();

    let mut rng = thread_rng();
    let mut w1 = Array2::random_using((n_features, hidden_size), StandardNormal, &mut rng);
    let mut b1 = Array2::zeros((1, hidden_size));
    let mut w2 = Array2::random_using((hidden_size, 1), StandardNormal, &mut rng);
    let mut b2 = Array2::zeros((1, 1));

    let mut losses = Vec::new();

    let mut final_pred = Array2::zeros((n_samples, 1));

    for epoch in 0..epochs {
        // Check if training should be stopped, only check for confirmed stop
        let should_stop = {
            let app_lock = app.lock().unwrap();
            let data_ref = app_lock.get_training_data();
            let data = data_ref.lock().unwrap();
            data.should_stop
        };
        
        if should_stop {
            println!("Training stopped early at epoch {}/{}", epoch, epochs);
            
            // Jika sudah ada beberapa epoch yang selesai, kita bisa menghitung akurasi
            if epoch > 0 {
                // Calculate final accuracy based on the current weights
                let z1 = x.dot(&w1) + &b1;
                let a1 = relu(&z1);
                let z2 = a1.dot(&w2) + &b2;
                let y_pred = sigmoid(&z2);
                
                let predictions = y_pred.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
                let correct = predictions
                    .iter()
                    .zip(y_true.iter())
                    .filter(|(p, y)| (*p - *y).abs() < 1e-6)
                    .count();
                let accuracy = (correct as f64 / n_samples as f64) * 100.0;
                
                // Mark training as completed with the current accuracy
                app.lock().unwrap().training_completed(accuracy);
                
                // Save the current loss plot
                if !losses.is_empty() {
                    plot_loss(&losses, epochs)?;
                }
            } else {
                // Jika belum ada epoch yang selesai, tandai sebagai tidak selesai
                let app_lock = app.lock().unwrap();
                let data_ref = app_lock.get_training_data();
                let mut data = data_ref.lock().unwrap();
                data.training_in_progress = false;
                data.completed = false;
            }
            
            return Ok(());
        }
        
        let z1 = x.dot(&w1) + &b1;
        let a1 = relu(&z1);
        let z2 = a1.dot(&w2) + &b2;
        let y_pred = sigmoid(&z2);

        let loss = binary_cross_entropy(&y_pred, &y_true);
        losses.push(loss);

        let dz2 = &y_pred - &y_true;
        let dw2 = a1.t().dot(&dz2) / n_samples as f64;
        let db2 = dz2.sum_axis(Axis(0)) / n_samples as f64;

        let da1 = dz2.dot(&w2.t());
        let dz1 = da1 * relu_deriv(&z1);
        let dw1 = x.t().dot(&dz1) / n_samples as f64;
        let db1 = dz1.sum_axis(Axis(0)) / n_samples as f64;

        w1 -= &(dw1 * learning_rate);
        b1 -= &(db1 * learning_rate);
        w2 -= &(dw2 * learning_rate);
        b2 -= &(db2 * learning_rate);

        final_pred = y_pred.clone();

        // Calculate accuracy periodically
        if epoch % LOG_INTERVAL == 0 || epoch == epochs - 1 {
            let predictions = y_pred.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
            let correct = predictions
                .iter()
                .zip(y_true.iter())
                .filter(|(p, y)| (*p - *y).abs() < 1e-6)
                .count();
            let accuracy = (correct as f64 / n_samples as f64) * 100.0;
            
            // Update progress with accuracy
            app.lock().unwrap().update_progress(epoch, loss, accuracy);
        } else {
            // Update progress without accuracy
            app.lock().unwrap().update_progress(epoch, loss, -1.0);
        }
        
        // Small sleep to give UI time to breathe
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    // Save loss plot to file
    plot_loss(&losses, epochs)?;

    // Calculate final accuracy
    let predictions = final_pred.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
    let correct = predictions
        .iter()
        .zip(y_true.iter())
        .filter(|(p, y)| (*p - *y).abs() < 1e-6)
        .count();

    let accuracy = (correct as f64 / n_samples as f64) * 100.0;
    
    // Mark training as completed
    app.lock().unwrap().training_completed(accuracy);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Ensure directories exist
    let csv_dir = std::path::Path::new("csv");
    let result_dir = std::path::Path::new("result");
    
    if !csv_dir.exists() {
        println!("Creating csv directory");
        std::fs::create_dir(csv_dir)?;
    }
    
    if !result_dir.exists() {
        println!("Creating result directory");
        std::fs::create_dir(result_dir)?;
    }
    
    // Create application options with a default window size
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 750.0]),
        ..Default::default()
    };
    
    // Create app
    let app = NeuralNetworkApp::new();
    
    // We need to move the app setup into the eframe creation callback
    // to ensure proper lifetimes
    eframe::run_native(
        "Neural Network Training",
        native_options,
        Box::new(|_cc| {
            // Clone the app and wrap in Arc<Mutex<>>
            let app_wrapped = Arc::new(Mutex::new(app));
            let app_clone = app_wrapped.clone();
            
            // Set up training callback
            {
                let mut app_locked = app_wrapped.lock().unwrap();
                
                // Set up the training callback but don't start training automatically
                app_locked.handle_train_click(move || {
                    let app_training = app_clone.clone();
                    
                    // Run training in a separate thread
                    thread::spawn(move || {
                        if let Err(e) = train_neural_network(app_training) {
                            eprintln!("Training error: {}", e);
                        }
                    });
                });
            }
            
            // Create a new instance before dropping the lock
            let app_ui = {
                let app_locked = app_wrapped.lock().unwrap();
                app_locked.clone()
            };
            
            Box::new(app_ui)
        }),
    )?;
    
    Ok(())
}