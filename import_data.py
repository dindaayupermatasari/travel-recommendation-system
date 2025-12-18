import pandas as pd
import mysql.connector
df = pd.read_csv("D:/Aulia/kuliah/TUGAS LAPRAK SMT 4/CODE/TWS/TRAVEL/data/data_travel.csv")
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='', 
    database='travel')
if conn.is_connected():
    print("Koneksi ke database berhasil.")
else:
    print("Gagal terhubung ke database.")
    exit()
cursor = conn.cursor()
for i, row in df.iterrows():
    try:
        values = tuple(row)
        cursor.execute("""
            INSERT INTO data_travel (gambar, kota, lokasi, nama_wisata, deskripsi, rating, ulasan, harga_asli, harga_diskon, cluster, label)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, values)
        print(f"Data baris {i} berhasil dimasukkan")
    except mysql.connector.Error as err:
        print(f"Error pada baris {i}: {err}")
conn.commit()
conn.close()
print("Proses selesai.")