use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Mutex;
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::ipv4::Ipv4Packet;
use pnet::packet::tcp::TcpPacket;
use pnet::packet::udp::UdpPacket;
use pnet::packet::Packet;
use std::net::Ipv4Addr;
use std::process::Command;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Set up iptables rules
    setup_iptables()?;

    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("Proxy server listening on 127.0.0.1:8080");

    let priority_queue = Arc::new(Mutex::new(Vec::new()));

    loop {
        let (client, _) = listener.accept().await?;
        let priority_queue = Arc::clone(&priority_queue);
        tokio::spawn(async move {
            if let Err(e) = handle_client(client, priority_queue).await {
                eprintln!("Error handling client: {}", e);
            }
        });
    }
}

async fn handle_client(mut client: TcpStream, priority_queue: Arc<Mutex<Vec<u8>>>) -> Result<(), Box<dyn Error>> {
    let mut buffer = [0; 8192];
    let n = client.read(&mut buffer).await?;

    let packet = &buffer[..n];
    let priority = process_packet(packet).await?;

    let mut queue = priority_queue.lock().await;
    queue.push(priority);

    // Forward the packet (in a real scenario, you'd connect to the destination here)
    client.write_all(packet).await?;

    Ok(())
}

async fn process_packet(packet: &[u8]) -> Result<u8, Box<dyn Error>> {
    if let Some(ipv4) = Ipv4Packet::new(packet) {
        match ipv4.get_next_level_protocol() {
            IpNextHeaderProtocols::Tcp => {
                if let Some(tcp) = TcpPacket::new(ipv4.payload()) {
                    // Extract features and get priority (placeholder)
                    Ok(get_priority_from_python(ipv4.get_source(), tcp.get_source(), tcp.get_destination()).await?)
                } else {
                    Ok(0) // Default priority
                }
            },
            IpNextHeaderProtocols::Udp => {
                if let Some(udp) = UdpPacket::new(ipv4.payload()) {
                    // Extract features and get priority (placeholder)
                    Ok(get_priority_from_python(ipv4.get_source(), udp.get_source(), udp.get_destination()).await?)
                } else {
                    Ok(0) // Default priority
                }
            },
            _ => Ok(0), // Default priority for other protocols
        }
    } else {
        Ok(0) // Default priority if not IPv4
    }
}

async fn get_priority_from_python(src_ip: Ipv4Addr, src_port: u16, dst_port: u16) -> Result<u8, Box<dyn Error>> {
    // Call Python script to get priority
    let output = Command::new("python")
        .arg("npu_model.py")
        .arg(src_ip.to_string())
        .arg(src_port.to_string())
        .arg(dst_port.to_string())
        .output()?;

    let priority = String::from_utf8(output.stdout)?.trim().parse()?;
    Ok(priority)
}

fn setup_iptables() -> Result<(), Box<dyn Error>> {
    Command::new("iptables")
        .args(&["-A", "OUTPUT", "-j", "NFQUEUE", "--queue-num", "1"])
        .status()?;
    Command::new("iptables")
        .args(&["-A", "INPUT", "-j", "NFQUEUE", "--queue-num", "1"])
        .status()?;
    Ok(())
}

fn cleanup_iptables() -> Result<(), Box<dyn Error>> {
    Command::new("iptables")
        .args(&["-D", "OUTPUT", "-j", "NFQUEUE", "--queue-num", "1"])
        .status()?;
    Command::new("iptables")
        .args(&["-D", "INPUT", "-j", "NFQUEUE", "--queue-num", "1"])
        .status()?;
    Ok(())
}