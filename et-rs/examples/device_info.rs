//! Print all fields returned by `Device::properties()`.
//!
//! ```bash
//! cargo run --manifest-path et-rs/Cargo.toml --release --example device_info
//! ```

fn main() -> et_soc1::Result<()> {
    let dev = et_soc1::Device::open(0)?;
    let p = dev.properties()?;

    println!("minion_boot_freq : {} MHz", p.minion_boot_freq);
    println!("shire_mask       : {:#x}", p.shire_mask);
    println!("cache_line_size  : {} B", p.cache_line_size);
    println!("total_l3_size    : {} KB", p.total_l3_size);
    println!("total_l2_size    : {} KB", p.total_l2_size);
    println!("total_scp_size   : {} KB", p.total_scp_size);
    println!("ddr_bandwidth    : {} MB/s", p.ddr_bandwidth);
    println!("num_l2_cache_banks: {}", p.num_l2_cache_banks);
    println!("sync_min_shire_id: {}", p.sync_min_shire_id);
    println!("form_factor      : {}", p.form_factor);
    println!("tdp              : {} W", p.tdp);
    println!("arch_rev         : {}", p.arch_rev);
    println!("devnum           : {}", p.devnum);

    assert_ne!(p.minion_boot_freq, 0, "minion_boot_freq should be non-zero");
    assert_ne!(p.shire_mask, 0, "shire_mask should be non-zero");
    println!("\nproperties OK");
    Ok(())
}
