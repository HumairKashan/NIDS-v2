use tokio::net::TcpStream;
use tokio::time::{sleep, Duration};
use tokio::io::AsyncWriteExt;
use serde::Serialize;

pub struct SocketSender {
    stream: TcpStream,
}

impl SocketSender {
    pub async fn new(addr: &str) -> anyhow::Result<Self> {
        let mut retries = 5;

        loop {
            match TcpStream::connect(addr).await {
                Ok(stream) => return Ok(Self { stream }),
                Err(e) => {
                    if retries == 0 {
                        return Err(e.into());
                    }
                    println!("[WARN] Python receiver not ready, retrying...");
                    sleep(Duration::from_secs(1)).await;
                    retries -= 1;
                }
            }
        }
    }

    pub async fn send_packet<T: Serialize>(&mut self, pkt: &T) -> anyhow::Result<()> {
        let json = serde_json::to_string(pkt)?;
        self.stream.write_all(json.as_bytes()).await?;
        self.stream.write_all(b"\n").await?;
        Ok(())
    }
} 