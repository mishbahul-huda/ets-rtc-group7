use std::io::{self, Write};

fn main() {
    // Membuat lookup table untuk sine dan cosine
    let size = 361; // 0-360 derajat
    let sine_table = create_sine_table(size, 1.0);
    let cosine_table = create_cosine_table(size, 1.0);

    let mut choice = String::new();

    loop {
        // Menampilkan menu
        println!(">>SINE AND COSINE LOOKUP TABLE BY GROUP 6<<");
        println!("Apa yang bisa saya bantu??");
        println!("1. Tampilkan seluruh tabel lookup (0-360 derajat)");
        println!("2. Masukkan sudut yang anda mau");
        println!("3. Exit, please press (P)");
        print!("Pilihan Anda: ");
        io::stdout().flush().unwrap();

        // Membaca input pengguna
        choice.clear();
        io::stdin().read_line(&mut choice).expect("Gagal membaca input");

        match choice.trim() {
            "1" => {
                display_full_table(&sine_table, &cosine_table);
            },
            "2" => {
                calculate_specific_angle(&sine_table, &cosine_table);
            },
            "P" | "p" => {
                println!("Terima kasih telah menggunakan program ini!");
                break;
            },
            _ => {
                println!("Pilihan tidak valid. Silakan coba lagi.");
            }
        }
    }
}

fn display_full_table(sine_table: &[f64], cosine_table: &[f64]) {
    println!("\n==== TABEL LOOKUP SINE DAN COSINE (0-360 DERAJAT) ====");
    println!("Sudut | Sine       | Cosine");
    println!("------|------------|------------");

    for i in 0..=360 {
        let sine_value = lookup_sine(sine_table, i as f64, 360.0);
        let cosine_value = lookup_cosine(cosine_table, i as f64, 360.0);

        println!("{:5}° | {:10.6} | {:10.6}", i, sine_value, cosine_value);

        // Memberikan jeda setiap 20 baris untuk memudahkan membaca
        if i % 20 == 0 && i > 0 {
            println!("Tekan Enter untuk melanjutkan...");
            let mut buffer = String::new();
            io::stdin().read_line(&mut buffer).expect("Gagal membaca input");
        }
    }
}

fn calculate_specific_angle(sine_table: &[f64], cosine_table: &[f64]) {
    loop {
        println!("\nMasukkan sudut dalam derajat (atau ketik 'kembali' untuk kembali ke menu utama):");
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Gagal membaca input");

        let input = input.trim();

        if input.to_lowercase() == "kembali" {
            break;
        }

        match input.parse::<f64>() {
            Ok(angle) => {
                let sine_value = lookup_sine(sine_table, angle, 360.0);
                let cosine_value = lookup_cosine(cosine_table, angle, 360.0);
                let rust_sine = angle.to_radians().sin();
                let rust_cosine = angle.to_radians().cos();

                println!("\nHasil perhitungan untuk sudut {:.2}°:", angle);
                println!("╔═══════════════╦════════════════╦════════════════╗");
                println!("║ Fungsi        ║ Lookup Table   ║ Fungsi Rust    ║");
                println!("╠═══════════════╬════════════════╬════════════════╣");
                println!("║ Sine          ║ {:14.8} ║ {:14.8} ║", sine_value, rust_sine);
                println!("║ Cosine        ║ {:14.8} ║ {:14.8} ║", cosine_value, rust_cosine);
                println!("╚═══════════════╩════════════════╩════════════════╝");

                // Menghitung dan menampilkan error
                // let sine_error = ((sine_value - rust_sine) / rust_sine * 100.0).abs();
                // let cosine_error = ((cosine_value - rust_cosine) / rust_cosine * 100.0).abs();

                // println!("Error Sine: {:.6}%", sine_error);
                // println!("Error Cosine: {:.6}%", cosine_error);
            },
            Err(_) => {
                println!("Input tidak valid. Masukkan angka numerik.");
            }
        }
    }
}

// Membuat lookup table untuk sine
fn create_sine_table(size: usize, period: f64) -> Vec<f64> {
    let mut table = vec![0.0; size];

    for i in 0..size {
        let angle = (i as f64 / (size - 1) as f64) * period * 2.0 * std::f64::consts::PI;
        table[i] = angle.sin();
    }

    table
}

// Membuat lookup table untuk cosine
fn create_cosine_table(size: usize, period: f64) -> Vec<f64> {
    let mut table = vec![0.0; size];

    for i in 0..size {
        let angle = (i as f64 / (size - 1) as f64) * period * 2.0 * std::f64::consts::PI;
        table[i] = angle.cos();
    }

    table
}

// Fungsi untuk mendapatkan nilai sine dari lookup table
fn lookup_sine(table: &[f64], angle_degrees: f64, period_degrees: f64) -> f64 {
    // Normalisasi sudut ke 0-period_degrees
    let normalized_angle = ((angle_degrees % period_degrees) + period_degrees) % period_degrees;

    // Konversi sudut ke indeks dalam tabel
    let table_size = table.len();
    let index_f = normalized_angle / period_degrees * (table_size - 1) as f64;

    // Interpolasi linear antara dua nilai terdekat dalam tabel
    let index_low = index_f.floor() as usize;
    let index_high = (index_low + 1).min(table_size - 1);

    let weight_high = index_f - index_f.floor();
    let weight_low = 1.0 - weight_high;

    table[index_low] * weight_low + table[index_high] * weight_high
}

// Fungsi untuk mendapatkan nilai cosine dari lookup table
    fn lookup_cosine(table: &[f64], angle_degrees: f64, period_degrees: f64) -> f64 {
        // Normalisasi sudut ke 0-period_degrees
        let normalized_angle = ((angle_degrees % period_degrees) + period_degrees) % period_degrees;

        // Konversi sudut ke indeks dalam tabel
        let table_size = table.len();
        let index_f = normalized_angle / period_degrees * (table_size - 1) as f64;

        // Interpolasi linear antara dua nilai terdekat dalam tabel
        let index_low = index_f.floor() as usize;
        let index_high = (index_low + 1).min(table_size - 1);

        let weight_high = index_f - index_f.floor();
        let weight_low = 1.0 - weight_high;

        table[index_low] * weight_low + table[index_high] * weight_high
}