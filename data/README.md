# Market Data (`data/`)

Thư mục này chứa file `prices.csv` gồm dữ liệu lịch sử theo ngày (OHLCV) của **5 cổ phiếu ngân hàng hàng đầu trên sàn HOSE (Việt Nam)**:

| Mã Yahoo Finance | Mã HOSE | Ngân hàng | Nhóm |
| :--- | :--- | :--- | :--- |
| `VCB.VN` | VCB | Vietcombank | Big4 (niêm yết) |
| `BID.VN` | BID | BIDV | Big4 (niêm yết) |
| `CTG.VN` | CTG | VietinBank | Big4 (niêm yết) |
| `MBB.VN` | MBB | MB Bank | Ngân hàng TMCP tư nhân top đầu |
| `ACB.VN` | ACB | ACB | Ngân hàng TMCP tư nhân top đầu |

> **Lưu ý:** Trong nhóm Big4 ngân hàng quốc doanh, Agribank hiện chưa niêm yết cổ phiếu trên sàn chứng khoán, vì vậy bộ dữ liệu gồm 3 ngân hàng Big4 đang niêm yết (`VCB`, `BID`, `CTG`) kết hợp cùng `MBB` và `ACB`.

## Cách tạo `data/prices.csv`

1. **Tải dữ liệu thật từ Yahoo Finance (khuyến nghị):**
   ```bash
   python scripts/download_data.py --years 5
   ```

2. **Tạo dữ liệu giả lập đa chế độ (chạy thử offline không cần mạng):**
   ```bash
   python scripts/download_data.py --synthetic --years 5
   ```

## Định dạng file `data/prices.csv`

Định dạng bảng chuẩn (long format), sắp xếp theo `(ticker, date)` tăng dần:

```csv
date,ticker,open,high,low,close,volume
2021-01-04,ACB.VN,15200.0,15500.0,15100.0,15450.0,12450000
...
```
