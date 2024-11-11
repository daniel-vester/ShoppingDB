import psycopg2
import ast


def insert_purchase_to_db(input_dict):
    default_value = -9999


    purchase_id = input_dict["purchase_id"] or default_value
    date = input_dict["date"] or default_value

    shop_data = ast.literal_eval(input_dict["shop"])
    try: shop_brand = shop_data[0]  
    except IndexError: shop_brand = default_value
    try:shop_name = shop_data[1] 
    except IndexError: shop_name = default_value
    try: shop_street = shop_data[2]
    except IndexError: shop_street = default_value
    try: shop_city = shop_data[3]
    except IndexError: shop_city= default_value


    

    conn = psycopg2.connect(database="db_shopping",
                        host="localhost",
                        user="functional_access",
                        password="1234",
                        port="5432")


    cursor = conn.cursor()

    for item in input_dict["items"]:
        try: item_name = item[0]
        except IndexError: item_name = default_value
        try: item_count = item[1]
        except IndexError: item_count = default_value
        try: item_price = item[2]
        except IndexError: item_price= default_value

        cursor.execute('''INSERT INTO RAW_RECEIPTS 
               (purchase_id, 
               date, 
               shop_brand, 
               shop_name, 
               shop_street, 
               shop_city,
               item_name,
               item_count,
               total_price
               ) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s)''', 
               (
                purchase_id,
                date, 
                shop_brand, 
                shop_name, 
                shop_street, 
                shop_city, 
                item_name,
                item_count,
                item_price))

        conn.commit()
    cursor.close()
    conn.close()