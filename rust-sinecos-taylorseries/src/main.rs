use std::io;

fn factorial(n: u32) -> f64 {
    (1..=n).fold(1.0, |acc, x| acc * (x as f64))
}

fn taylor_sin(x: f64, terms: u32) -> f64 {
    let mut sum = 0.0;
    for n in 0..terms {
        let term = ((-1.0f64).powi(n as i32) * x.powi((2 * n + 1) as i32)) / factorial(2 * n + 1);
        sum += term;
    }
    sum
}

fn taylor_cos(x: f64, terms: u32) -> f64 {
    let mut sum = 0.0;
    for n in 0..terms {
        let term = ((-1.0f64).powi(n as i32) * x.powi((2 * n) as i32)) / factorial(2 * n);
        sum += term;
    }
    sum
}

fn main() {
    let terms = 10; // Jumlah suku Taylor yang digunakan

    loop {
        let mut input = String::new();
        println!("\nMasukkan sudut dalam derajat (atau ketik 'exit' untuk keluar):");
        io::stdin().read_line(&mut input).expect("Gagal membaca input");

        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") {
            println!("Terima kasih! Program selesai.");
            break;
        }

        let angle_deg: f64 = match input.parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Harap masukkan angka yang valid!");
                continue;
            }
        };

        let angle_rad = angle_deg.to_radians(); // Konversi derajat ke radian

        println!("sin({}°) ≈ {}", angle_deg, taylor_sin(angle_rad, terms));
        println!("cos({}°) ≈ {}", angle_deg, taylor_cos(angle_rad, terms));
    }
}
