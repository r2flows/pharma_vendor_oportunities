import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Análisis de Compras y Productos POS", layout="wide")
st.title("Análisis de Compras Reales vs Potenciales por Punto de Venta")

# Funciones de utilidad
def get_status_description(status):
    """
    Convierte un código de status numérico en su descripción correspondiente
    
    Args:
        status: Código de status (1: activo, 2: pendiente, 0: rechazado) o np.nan
        
    Returns:
        Descripción del status
    """
    if pd.isna(status): 
        return "Sin Status"
    
    status_map = {
        0: "Rechazado", 
        1: "Activo", 
        2: "Pendiente"
    }
    
    return status_map.get(status, f"Status {status}")

def obtener_status_vendor(vendor_id, pos_id, df_vendors_pos):
    """
    Obtiene el status de un vendor para un punto de venta específico
    
    Args:
        vendor_id: ID del vendor
        pos_id: ID del punto de venta
        df_vendors_pos: DataFrame con relaciones vendor-pos
        
    Returns:
        Status del vendor (1: activo, 2: pendiente, 0: rechazado) o np.nan si no existe relación
    """
    # Asegurarse de que vendor_id y point_of_sale_id sean numéricos para comparación correcta
    vendor_id = pd.to_numeric(vendor_id, errors='coerce')
    pos_id = pd.to_numeric(pos_id, errors='coerce')
    
    # Filtrar el DataFrame para obtener la relación específica
    if 'vendor_id' in df_vendors_pos.columns and 'point_of_sale_id' in df_vendors_pos.columns and 'status' in df_vendors_pos.columns:
        # Asegurar que los tipos de datos sean numéricos en el DataFrame
        df_vendors_pos_copy = df_vendors_pos.copy()
        df_vendors_pos_copy['vendor_id'] = pd.to_numeric(df_vendors_pos_copy['vendor_id'], errors='coerce')
        df_vendors_pos_copy['point_of_sale_id'] = pd.to_numeric(df_vendors_pos_copy['point_of_sale_id'], errors='coerce')
        
        # Buscar la relación
        relacion = df_vendors_pos_copy[
            (df_vendors_pos_copy['vendor_id'] == vendor_id) & 
            (df_vendors_pos_copy['point_of_sale_id'] == pos_id)
        ]
        
        # Si se encuentra la relación, devolver el status
        if not relacion.empty:
            return relacion['status'].iloc[0]
    
    # Si no se encuentra la relación, devolver np.nan
    return np.nan

def obtener_geo_zone(address):
    partes = address.split(', ')
    return ', '.join(partes[-2:-1])

def unificar_productos_sin_duplicados(df_global, df_local):
    """Unifica productos con precios mínimos sin duplicados, priorizando productos locales"""
    if df_global.empty and df_local.empty: return pd.DataFrame()
    if df_global.empty: return df_local.copy()
    if df_local.empty: return df_global.copy()
    
    df_global_copy, df_local_copy = df_global.copy(), df_local.copy()
    df_global_copy['origen'] = 'global'
    df_local_copy['origen'] = 'local'
    
    df_combinado = pd.concat([df_local_copy, df_global_copy], ignore_index=True)
    
    subset_cols = []
    for col in ['point_of_sale_id', 'super_catalog_id', 'vendor_id']:
        if col in df_combinado.columns:
            subset_cols.append(col)
    
    if not subset_cols: return df_combinado
    
    df_combinado = df_combinado.sort_values('origen', ascending=False)
    df_unificado = df_combinado.drop_duplicates(subset=subset_cols, keep='first')
    
    return df_unificado

def load_vendors_dm():
    """Carga y procesa el archivo vendors_dm.csv"""
    try:
        df_vendor_dm = pd.read_csv('vendors_dm.csv')
        # Asegurarse de que las columnas estén correctamente nombradas
        if 'client_id' in df_vendor_dm.columns and 'vendor_id' not in df_vendor_dm.columns:
            df_vendor_dm.rename(columns={'client_id': 'vendor_id'}, inplace=True)
        return df_vendor_dm
    except Exception as e:
        print(f"Error al procesar vendors_dm.csv: {e}")
        return pd.DataFrame(columns=['vendor_id', 'name', 'drug_manufacturer_id'])

def crear_dataframe_vendors_dm(detail_table, df_vendor_dm):
    """
    Crea un dataframe con los vendors que también son drug manufacturers
    
    Args:
        detail_table: DataFrame con información de compras por vendor
        df_vendor_dm: DataFrame con relaciones vendor-drug_manufacturer
    
    Returns:
        DataFrame filtrado solo con vendors que son drug manufacturers
    """
    if detail_table.empty or df_vendor_dm.empty:
        return pd.DataFrame()
    
    # CAMBIO CLAVE: Obtener la lista de drug_manufacturer_ids en lugar de vendor_ids
    dm_ids = set(df_vendor_dm['drug_manufacturer_id'].unique())
    
    # Convertir a tipo numérico para comparación adecuada
    detail_table['Droguería/Vendor ID'] = pd.to_numeric(detail_table['Droguería/Vendor ID'], errors='coerce')
    
    # Filtrar la tabla de detalle para incluir vendors donde "Droguería/Vendor ID" coincide con algún drug_manufacturer_id
    dm_vendors_detail = detail_table[detail_table['Droguería/Vendor ID'].isin(dm_ids)].copy()
    
    if not dm_vendors_detail.empty:
        # Añadir columna que indique que son drug manufacturers
        #dm_vendors_detail['Es Drug Manufacturer'] = 'Sí'
        
        # Para cada fila en dm_vendors_detail, encontrar el vendor_id correspondiente
        dm_vendors_detail['Vendor Real ID'] = None
        for idx, row in dm_vendors_detail.iterrows():
            dm_id = row['Droguería/Vendor ID']
            vendor_matches = df_vendor_dm[df_vendor_dm['drug_manufacturer_id'] == dm_id]
            if not vendor_matches.empty:
                dm_vendors_detail.at[idx, 'Vendor Real ID'] = vendor_matches.iloc[0]['vendor_id']
    
    return dm_vendors_detail
    
    return pd.DataFrame()

def calcular_potencial_convertido(df_pedidos, df_vendor_dm):
    """Calcula el potencial convertido basado en compras reales de vendors que son drug manufacturers"""
    if df_pedidos.empty or df_vendor_dm.empty: 
        return pd.DataFrame()
    
    if 'vendor_id' not in df_pedidos.columns or 'vendor_id' not in df_vendor_dm.columns:
        print("Faltan columnas para calcular potencial convertido")
        return pd.DataFrame()
    
    df_pedidos_dm = pd.merge(df_pedidos, df_vendor_dm[['vendor_id']], on='vendor_id', how='inner')
    
    if df_pedidos_dm.empty:
        return pd.DataFrame()
    
    if 'point_of_sale_id' in df_pedidos_dm.columns and 'vendor_id' in df_pedidos_dm.columns:
        if 'total_compra' not in df_pedidos_dm.columns and 'unidades_pedidas' in df_pedidos_dm.columns and 'precio_minimo' in df_pedidos_dm.columns:
            df_pedidos_dm['total_compra'] = df_pedidos_dm['unidades_pedidas'] * df_pedidos_dm['precio_minimo']
        
        if 'total_compra' in df_pedidos_dm.columns:
            pot_convertido = df_pedidos_dm.groupby(['point_of_sale_id', 'vendor_id'])['total_compra'].sum().reset_index()
            pot_convertido.columns = ['point_of_sale_id', 'vendor_id', 'valor_convertido']
            return pot_convertido
    
    return pd.DataFrame()

def create_simple_summary(df_products, df_local_products=None, orders_total=0, products_total=0, local_products_total=0):
    """Crea un DataFrame resumen con la información de potencial y ahorro"""
    if df_products.empty: return pd.DataFrame()
    
    pos_totals = df_products.groupby('point_of_sale_id')['valor_total_vendedor'].sum().to_dict()
    
    # Agregar valores de productos locales
    if df_local_products is not None and not df_local_products.empty:
        for pos_id in df_local_products['point_of_sale_id'].unique():
            pos_local_total = df_local_products[df_local_products['point_of_sale_id'] == pos_id]['valor_total_vendedor'].sum()
            pos_totals[pos_id] = pos_totals.get(pos_id, 0) + pos_local_total
    
    # Calcular ahorro global
    combined_total = products_total + local_products_total
    savings_percentage = ((orders_total - combined_total) / orders_total * 100) if orders_total > 0 else 0
    
    summary_data = []
    
    # Procesar productos globales
    if not df_products.empty:
        grouped = df_products.groupby(['point_of_sale_id', 'vendor_id'])
        for (pos_id, vendor_id), group in grouped:
            total_pos = pos_totals.get(pos_id, 0)
            if total_pos > 0:
                summary_data.append({
                    'point_of_sale_id': pos_id,
                    'vendor_id': vendor_id,
                    'status': group['status'].iloc[0] if 'status' in group.columns else np.nan,
                    'valor_potencial': group['valor_total_vendedor'].sum(),
                    'tipo_oportunidad': 'Global',
                    'porcentaje_ahorro': (group['valor_total_vendedor'].sum() / total_pos) * 100 * savings_percentage/100
                })
    
    # Procesar productos locales
    if df_local_products is not None and not df_local_products.empty:
        grouped_local = df_local_products.groupby(['point_of_sale_id', 'vendor_id'])
        for (pos_id, vendor_id), group in grouped_local:
            total_pos = pos_totals.get(pos_id, 0)
            if total_pos > 0:
                # Verificar si ya existe esta combinación
                existing_entry = next((item for item in summary_data if 
                                      item['point_of_sale_id'] == pos_id and 
                                      item['vendor_id'] == vendor_id), None)
                
                if existing_entry:
                    # Actualizar entrada existente
                    existing_entry['valor_potencial'] += group['valor_total_vendedor'].sum()
                    existing_entry['tipo_oportunidad'] = 'Global y Local'
                    existing_entry['porcentaje_ahorro'] = (existing_entry['valor_potencial'] / total_pos) * 100 * savings_percentage/100
                else:
                    # Crear nueva entrada
                    status = group['status'].iloc[0] if 'status' in group.columns else np.nan
                    summary_data.append({
                        'point_of_sale_id': pos_id,
                        'vendor_id': vendor_id,
                        'status': status,
                        'valor_potencial': group['valor_total_vendedor'].sum(),
                        'tipo_oportunidad': 'Local',
                        'porcentaje_ahorro': (group['valor_total_vendedor'].sum() / total_pos) * 100 * savings_percentage/100
                    })
    
    # Crear DataFrame final y ordenar
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(['point_of_sale_id', 'valor_potencial'], ascending=[True, False])
    
    return summary_df

def mostrar_tabla_vendor_detalle(vendor_df, dm_vendors_detail):
    """
    Función para mostrar la tabla detallada de vendors con manejo seguro de columnas
    """
    if vendor_df.empty:
        st.info("No hay datos disponibles para mostrar.")
        return
    
    # Extraer columnas básicas - Verificar la existencia de cada columna antes de usarla
    display_columns = []
    for col in ['Vendor ID', 'Status', 'Valor Potencial Total', 'Valor Convertido']:
        if col in vendor_df.columns:
            display_columns.append(col)
    
    # Asegurarse de que hay columnas para mostrar
    if not display_columns:
        st.warning("No hay columnas válidas para mostrar.")
        return
    
    # Crear display_df solo con las columnas disponibles
    display_df = vendor_df[display_columns].copy()
    
    # Asegurarse de que 'Total Comprado Como DM' esté presente (añadir si no existe)
    if 'Total Comprado Como DM' not in display_df.columns:
        display_df['Total Comprado Como DM'] = 0.0
    
    # Crear una copia de display_df para no modificar el original
    display_df_combined = display_df.copy()
    
    # Si dm_vendors_detail tiene datos y la columna necesaria, procesar
    if not dm_vendors_detail.empty and 'Vendor Real ID' in dm_vendors_detail.columns:
        try:
            # Convertir las columnas de ID a numérico para asegurar una correcta comparación
            display_df_combined['Vendor ID'] = pd.to_numeric(display_df_combined['Vendor ID'], errors='coerce')
            dm_vendors_detail['Vendor Real ID'] = pd.to_numeric(dm_vendors_detail['Vendor Real ID'], errors='coerce')
            
            # Crear un diccionario de drug manufacturers para facilitar el lookup
            dm_dict = {}
            for _, row in dm_vendors_detail.iterrows():
                if pd.notna(row['Vendor Real ID']):
                    dm_dict[row['Vendor Real ID']] = {
                        'Total Comprado Como DM': row.get('Total Comprado', 0)
                    }
            
            # Rellenar información de Total Comprado Como DM
            for idx, row in display_df_combined.iterrows():
                vendor_id = row['Vendor ID']
                if vendor_id in dm_dict:
                    display_df_combined.at[idx, 'Total Comprado Como DM'] = dm_dict[vendor_id]['Total Comprado Como DM']
        except Exception as e:
            st.warning(f"Error al procesar datos de drug manufacturers: {str(e)}")
    
    # Mostrar el dataframe combinado con manejo seguro de columnas
    try:
        # Asegurarse de que todas las columnas necesarias existan
        required_cols = [
            'Vendor ID', 'Status', 'Valor Potencial Total', 'Valor Convertido', 
            'Total Comprado Como DM'
        ]
        
        for col in required_cols:
            if col not in display_df_combined.columns:
                display_df_combined[col] = 0.0 if col in ['Valor Potencial Total', 'Valor Convertido', 'Total Comprado Como DM'] else (
                    'Sin Status' if col == 'Status' else ''
                )
        
        # Aplicar formato
        styled_df = display_df_combined.style.format({
            'Valor Potencial Total': '${:,.2f}',
            'Valor Convertido': '${:,.2f}',
            'Total Comprado Como DM': '${:,.2f}'
        })
        
        # Aplicar estilos condicionales
        styled_df = styled_df.applymap(
            lambda x: 'background-color: #90EE90' if x == "Activo" else 
            ('background-color: #FFD700' if x == "Pendiente" else 
             'background-color: #ffcccb' if x == "Sin Status" else ''),
            subset=['Status']
        )
        
        st.dataframe(styled_df)
    except Exception as e:
        # Mostrar versión simplificada en caso de error
        st.warning(f"Error al aplicar formato avanzado: {str(e)}")
        st.dataframe(display_df_combined)

@st.cache_data
def load_and_process_data():
    """Función principal que procesa todos los datos necesarios"""
    try:
        # Cargar archivos
        df_pos_address = pd.read_csv('pos_address.csv')
        df_pedidos = pd.read_csv('orders_delivered_pos_vendor_geozone.csv')
        df_proveedores = pd.read_csv('vendors_catalog.csv')
        df_vendors_pos = pd.read_csv('vendor_pos_relations.csv')
        df_products = pd.read_csv('top_5_productos_geozona.csv')
        df_vendor_dm = load_vendors_dm()
        #st.dataframe(df_vendors_pos.count())
        try:
            df_min_purchase = pd.read_csv('minimum_purchase.csv')
        except FileNotFoundError:
            df_min_purchase = pd.DataFrame(columns=['vendor_id', 'name', 'min_purchase'])
        
        # Procesar dirección y geo_zone
        df_pos_address['geo_zone'] = df_pos_address['address'].apply(obtener_geo_zone)
        
        if 'geo_zone' in df_pedidos.columns:
            df_pedidos = df_pedidos.drop(columns=['geo_zone'])
            
        # Normalizar datos
        df_proveedores['percentage'].fillna(0, inplace=True)
        pos_geo_zones = df_pos_address[['point_of_sale_id', 'geo_zone']]
        #st.write(pos_geo_zones)
        # Reemplazar abreviaturas
        abreviaturas = {
            'B.C.S.': 'Baja California Sur', 'Qro.': 'Querétaro', 'Jal.': 'Jalisco',
            'Pue.': 'Puebla', 'Méx.': 'CDMX', 'Oax.': 'Oaxaca', 'Chih.': 'Chihuahua',
            'Coah.': 'Coahuila de Zaragoza', 'Mich.': 'Michoacán de Ocampo',
            'Ver.': 'Veracruz de Ignacio de la Llave', 'Chis.': 'Chiapas',
            'N.L.': 'Nuevo León', 'Hgo.': 'Hidalgo', 'Tlax.': 'Tlaxcala',
            'Tamps.': 'Tamaulipas', 'Yuc.': 'Yucatan', 'Mor.': 'Morelos',
            'Sin.': 'Sinaloa', 'S.L.P.': 'San Luis Potosí', 'Q.R.': 'Quintana Roo',
            'Dgo.': 'Durango', 'B.C.': 'Baja California', 'Gto.': 'Guanajuato',
            'Camp.': 'Campeche', 'Tab.': 'Tabasco', 'Son.': 'Sonora',
            'Gro.': 'Guerrero', 'Zac.': 'Zacatecas', 'Ags.': 'Aguascalientes',
            'Nay.': 'Nayarit'
        }
        pos_geo_zones['geo_zone'] = pos_geo_zones['geo_zone'].replace(abreviaturas)
        
        # Separar proveedores nacionales y regionales
        df_proveedores_nacional = df_proveedores[df_proveedores['name'] == 'México'].copy()
        df_proveedores_regional = df_proveedores[df_proveedores['name'] != 'México'].copy()
        
        # Unir pedidos con zonas geográficas
        df_pedidos_zonas = pd.merge(df_pedidos, pos_geo_zones, on='point_of_sale_id', how='left')
        df_pedidos_zonas = df_pedidos_zonas[df_pedidos_zonas['unidades_pedidas'] > 0]
        
        # Asegurar nombre de columna correcto
        #if 'droguería' in df_pedidos_zonas.columns and 'vendor_id' not in df_pedidos_zonas.columns:
         #   df_pedidos_zonas.rename(columns={'droguería': 'vendor_id'}, inplace=True)
        
        # Procesar con proveedores nacionales y regionales
        df_pedidos_proveedores_nacional = pd.merge(
            df_pedidos_zonas, df_proveedores_nacional, on='super_catalog_id', how='inner'
        )
        df_pedidos_proveedores_nacional = df_pedidos_proveedores_nacional[
            df_pedidos_proveedores_nacional['unidades_pedidas'] > 0
        ]
        
        df_pedidos_proveedores_regional = pd.merge(
            df_pedidos_zonas, df_proveedores_regional, 
            left_on=['super_catalog_id', 'geo_zone'], right_on=['super_catalog_id', 'name'], 
            how='inner'
        )
        df_pedidos_proveedores_regional = df_pedidos_proveedores_regional[
            df_pedidos_proveedores_regional['unidades_pedidas'] > 0
        ]
        
        df_pedidos_proveedores_nacional['base_price'] = df_pedidos_proveedores_nacional['base_price'].astype(float)
#df_proveedores['precio_vendedor']=df_proveedores['precio_vendedor'].astype(float)
        df_pedidos_proveedores_nacional['percentage'] = df_pedidos_proveedores_nacional['percentage'].astype(float)

        df_pedidos_proveedores_regional['base_price'] = df_pedidos_proveedores_regional['base_price'].astype(float)
#df_proveedores['precio_vendedor']=df_proveedores['precio_vendedor'].astype(float)
        df_pedidos_proveedores_regional['percentage'] = df_pedidos_proveedores_regional['percentage'].astype(float)


        df_pedidos_proveedores_nacional['precio_vendedor'] = df_pedidos_proveedores_nacional['base_price'] + (df_pedidos_proveedores_nacional['base_price'] * df_pedidos_proveedores_nacional['percentage'] / 100)
        df_pedidos_proveedores_regional['precio_vendedor'] = df_pedidos_proveedores_regional['base_price'] + (df_pedidos_proveedores_regional['base_price'] * df_pedidos_proveedores_regional['percentage'] / 100)

        # Unir dataframes
        df_pedidos_proveedores = pd.concat([
            df_pedidos_proveedores_regional, df_pedidos_proveedores_nacional
        ], axis=0, ignore_index=True)
        
        # Verificar y corregir columnas
       # alternative_vendor_columns = ['droguería', 'vendor_id_x', 'vendor_id_y']
        #if 'vendor_id' not in df_pedidos_proveedores.columns:
         #   for alt_col in alternative_vendor_columns:
          #      if alt_col in df_pedidos_proveedores.columns:
           #         df_pedidos_proveedores.rename(columns={alt_col: 'vendor_id'}, inplace=True)
            #        break
        
        # Definir columnas para eliminar duplicados
        subset_columns = [col for col in ['super_catalog_id', 'vendor_id', 'geo_zone'] 
                          if col in df_pedidos_proveedores.columns]
        
        # Eliminar duplicados
        df_pedidos_proveedores_unique = df_pedidos_proveedores.drop_duplicates(subset=['super_catalog_id','vendor_id_y','geo_zone'])
        
        # Calcular precio_total_vendedor
        if 'precio_vendedor' in df_pedidos_proveedores_unique.columns and 'unidades_pedidas' in df_pedidos_proveedores_unique.columns:
            df_pedidos_proveedores_unique['precio_total_vendedor'] = (
                df_pedidos_proveedores_unique['unidades_pedidas'].astype(float) * 
                df_pedidos_proveedores_unique['precio_vendedor'].astype(float)
            )
        
        # Unir con relaciones vendor-pos
        if 'vendor_id' in df_pedidos_proveedores_unique.columns and 'point_of_sale_id' in df_pedidos_proveedores_unique.columns:
            df_pedidos_proveedores_unique = pd.merge(
                df_pedidos_proveedores_unique, df_vendors_pos,
                on=['point_of_sale_id', 'vendor_id'], how='left'
            )
        #st.dataframe(df_pedidos_proveedores_unique.count())
        
        # Corregir nombres de columnas si es necesario
        #---------------------#
        #for col_pair in [
         #   ('point_of_sale_id_x', 'point_of_sale_id'),
          #  ('super_catalog_id_x', 'super_catalog_id'),
           # ('precio_minimo_x', 'precio_minimo')
        #]:
         #   if col_pair[0] in df_pedidos_proveedores_unique.columns and col_pair[1] not in df_pedidos_proveedores_unique.columns:
          #      df_pedidos_proveedores_unique.rename(columns={col_pair[0]: col_pair[1]}, inplace=True)
        
        # Calcular precios mínimos locales
        df_pedidos_proveedores_unique.rename(columns={'vendor_id':'drug_manufacturer_id', 'vendor_id_y':'vendor_id'}, inplace=True)
        cols_needed = ['point_of_sale_id', 'super_catalog_id', 'precio_minimo']
        if all(col in df_pedidos_proveedores_unique.columns for col in cols_needed):
            min_prices = (df_pedidos_proveedores_unique
                         .groupby(['point_of_sale_id', 'super_catalog_id'])['precio_minimo']
                         .min()
                         .reset_index())
            min_prices.columns = ['point_of_sale_id', 'super_catalog_id', 'precio_minimo_orders']
            
            # Unir para comparar precios
            df_con_precios_minimos_local = pd.merge(
                df_pedidos_proveedores_unique, min_prices,
                on=['point_of_sale_id', 'super_catalog_id'], how='left'
            )
            # Identificar productos ganadores locales
            if 'precio_vendedor' in df_con_precios_minimos_local.columns:
                df_productos_ganadores_local = df_con_precios_minimos_local[
                    df_con_precios_minimos_local['precio_minimo_orders'] > df_con_precios_minimos_local['precio_vendedor']
                ]
                
                # Asegurar que tenga valor_total_vendedor
                if 'valor_total_vendedor' not in df_productos_ganadores_local.columns and 'unidades_pedidas' in df_productos_ganadores_local.columns:
                    df_productos_ganadores_local['valor_total_vendedor'] = (
                        df_productos_ganadores_local['unidades_pedidas'].astype(float) * 
                        df_productos_ganadores_local['precio_vendedor'].astype(float)
                    )
            else:
                df_productos_ganadores_local = pd.DataFrame()
        #### FILTRO PRUEBA PRODUCTOS GANADORES
        #st.dataframe(df_productos_ganadores_local[(df_productos_ganadores_local['vendor_id_x']==10269) & (df_productos_ganadores_local['point_of_sale_id'] == 1409)].drop_duplicates('super_catalog_id'))        # Unificar productos globales y locales sin duplicados

        #else:
        #    df_productos_ganadores_local = pd.DataFrame()
        df_productos_unificados = unificar_productos_sin_duplicados(df_products, df_productos_ganadores_local)
        
        # Calcular potencial convertido
        df_potencial_convertido = calcular_potencial_convertido(df_pedidos, df_vendor_dm)
        
        # Calcular métricas para visualización
        df_orders = df_pedidos.copy()
        
        # Agregar total_compra si no existe
        if 'total_compra' not in df_orders.columns and 'unidades_pedidas' in df_orders.columns and 'precio_minimo' in df_orders.columns:
            df_orders['total_compra'] = df_orders['unidades_pedidas'] * df_orders['precio_minimo']
        
        # Calcular estadísticas por POS
        if all(col in df_orders.columns for col in ['point_of_sale_id', 'order_id', 'total_compra']):
            order_totals = df_orders.groupby(['point_of_sale_id', 'order_id'])['total_compra'].sum().reset_index()
            pos_order_stats = order_totals.groupby('point_of_sale_id').agg({
                'total_compra': ['mean', 'count']
            }).reset_index()
            pos_order_stats.columns = ['point_of_sale_id', 'promedio_por_orden', 'numero_ordenes']
        else:
            pos_order_stats = pd.DataFrame(columns=['point_of_sale_id', 'promedio_por_orden', 'numero_ordenes'])
        
        # Calcular totales por vendor
        if all(col in df_orders.columns for col in ['point_of_sale_id', 'vendor_id', 'total_compra']):
            pos_vendor_totals = df_orders.groupby(['point_of_sale_id', 'vendor_id'])['total_compra'].sum().reset_index()
        else:
            pos_vendor_totals = pd.DataFrame(columns=['point_of_sale_id', 'vendor_id', 'total_compra'])
        
        return pos_vendor_totals, df_pedidos, df_productos_unificados, pos_order_stats, df_min_purchase, df_potencial_convertido, df_vendor_dm, pos_geo_zones
    
    except Exception as e:
        import traceback
        print("Error en load_and_process_data:", traceback.format_exc())
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df, empty_df, empty_df, empty_df, empty_df, empty_df

def crear_grafico_oportunidades(vendor_df, df_potencial_convertido, selected_pos, dm_vendors_detail=None):
    """Crea gráfico con potencial, potencial convertido y valores comprados como DM"""
    if vendor_df.empty:
        return go.Figure()
    
    # Ordenar por potencial total
    vendor_df_sorted = vendor_df.sort_values('Valor Potencial Total', ascending=True)
    
    fig = go.Figure()
    
    # Preparar diccionario de valores comprados como DM
    dm_values_dict = {}
    if dm_vendors_detail is not None and not dm_vendors_detail.empty and 'Vendor Real ID' in dm_vendors_detail.columns:
        # Convertir a tipos numéricos para comparación correcta
        dm_vendors_detail['Vendor Real ID'] = pd.to_numeric(dm_vendors_detail['Vendor Real ID'], errors='coerce')
        
        # Crear diccionario de valores DM
        for _, row in dm_vendors_detail.iterrows():
            if pd.notna(row.get('Vendor Real ID')):
                vendor_id = row['Vendor Real ID']
                dm_values_dict[vendor_id] = row.get('Total Comprado', 0)
    
    # Añadir barra para potencial
    fig.add_trace(go.Bar(
        name='Potencial',
        x=vendor_df_sorted['Valor Potencial Total'],
        y=[str(int(vid)) for vid in vendor_df_sorted['Vendor ID']],
        orientation='h',
        marker_color='rgb(26, 118, 255)'
    ))
    
    # Añadir barra para valor convertido (directamente del dataframe vendor_df)
    valores_convertidos = vendor_df_sorted['Valor Convertido'].tolist()
    
    # Solo añadir la barra si hay al menos un valor mayor que cero
    if any(val > 0 for val in valores_convertidos):
        fig.add_trace(go.Bar(
            name='Valor Convertido',
            x=valores_convertidos,
            y=[str(int(vid)) for vid in vendor_df_sorted['Vendor ID']],
            orientation='h',
            marker_color='rgb(55, 183, 109)'  # Verde para valor convertido
        ))
    
    # Añadir barra para valores comprados como DM
    if dm_values_dict:
        # Obtener valores DM para cada vendor en el mismo orden que vendor_df_sorted
        dm_values = []
        for vid in vendor_df_sorted['Vendor ID']:
            dm_values.append(dm_values_dict.get(vid, 0))
        
        # Solo añadir la barra si hay al menos un valor mayor que cero
        if any(val > 0 for val in dm_values):
            fig.add_trace(go.Bar(
                name='Comprado como DM',
                x=dm_values,
                y=[str(int(vid)) for vid in vendor_df_sorted['Vendor ID']],
                orientation='h',
                marker_color='rgb(255, 127, 14)'  # Color naranja para destacar
            ))
    
    # Configurar layout
    fig.update_layout(
        title='Análisis de Potencial, Valor Convertido y Compras DM por Vendor',
        xaxis_title='Valor ($)',
        yaxis_title='Vendor ID',
        barmode='group',  # Mantener barras agrupadas
        height=max(500, len(vendor_df) * 40),
        yaxis={
            'type': 'category',
            'tickmode': 'array',
            'ticktext': [str(int(vid)) for vid in vendor_df_sorted['Vendor ID']],
            'tickvals': list(range(len(vendor_df_sorted))),
        },
        margin=dict(l=150, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Formato monetario
    fig.update_xaxes(tickformat='$,.0f')
    
    return fig


# Cargar datos
try:    
    pos_vendor_totals, df_original, df_productos_unificados, pos_order_stats, df_min_purchase, df_potencial_convertido, df_vendor_dm, pos_geo_zones = load_and_process_data()
    
    # Cargar el archivo vendors_dm.csv
    df_vendor_dm = pd.DataFrame()
    try:
        df_vendor_dm = pd.read_csv('vendors_dm.csv')
        # Asegurarse de que las columnas estén correctamente nombradas
        if 'client_id' in df_vendor_dm.columns and 'vendor_id' not in df_vendor_dm.columns:
            df_vendor_dm.rename(columns={'client_id': 'vendor_id'}, inplace=True)
    except Exception as e:
        print(f"Error al cargar vendors_dm.csv: {e}")

# Cargar el archivo vendor_pos_relations.csv
    df_vendors_pos = pd.DataFrame()
    try:
        df_vendors_pos = pd.read_csv('vendor_pos_relations.csv')
    except Exception as e:
        print(f"Error al cargar vendor_pos_relations.csv: {e}")
        st.warning("No se pudo cargar la información de relaciones vendor-pos. Algunas funcionalidades podrían estar limitadas.")


    # Filtro de punto de venta
    st.header("Análisis Individual de POS")
    pos_list = sorted(list(set(pos_vendor_totals['point_of_sale_id']))) if not pos_vendor_totals.empty else []
    
    if not pos_list:
        st.warning("No hay puntos de venta disponibles para analizar")
    else:
        selected_pos = st.selectbox("Seleccionar Punto de Venta", options=pos_list)

        # Mostrar información del POS seleccionado
        if selected_pos:
            # Filtrar datos para el POS seleccionado
            pos_data = pos_vendor_totals[pos_vendor_totals['point_of_sale_id'] == selected_pos]
            pos_data = pos_data.sort_values('total_compra', ascending=False) if not pos_data.empty else pd.DataFrame()

            # Obtener estadísticas
            pos_stats = pos_order_stats[pos_order_stats['point_of_sale_id'] == selected_pos]
            promedio_por_orden = pos_stats.iloc[0]['promedio_por_orden'] if not pos_stats.empty else 0
            numero_ordenes = int(pos_stats.iloc[0]['numero_ordenes']) if not pos_stats.empty else 0
                
            st.subheader("Información del Punto de Venta")

            # Métricas principales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Total de Compras - POS {selected_pos}", 
                          f"${pos_data['total_compra'].sum():,.2f}" if not pos_data.empty else "$0.00")
            with col2:
                st.metric("Promedio por Orden", f"${promedio_por_orden:,.2f}")
            with col3:
                st.metric("Número de Órdenes", f"{numero_ordenes:,}")

            # Información adicional
            pos_info = pos_geo_zones[pos_geo_zones['point_of_sale_id'] == selected_pos]
            pos_country = df_original[df_original['point_of_sale_id'] == selected_pos]

            country = pos_country['country'].iloc[0] if not pos_country.empty and 'country' in pos_country.columns else 'No disponible'
            geo_zone = pos_info['geo_zone'].iloc[0] if not pos_info.empty and 'geo_zone' in pos_info.columns else 'No disponible'

            #geo_zone = ''

           # if not pos_info.empty and 'geo_zone' in pos_info.columns:
        # Usar el primer valor no nulo de geo_zone
            #    geo_zones = pos_info['geo_zone'].dropna().unique()
             #   if len(geo_zones) > 0:
              #      geo_zone = geo_zones[0]


            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("País", country)
            with info_col2:
                st.metric("Zona Geográfica", geo_zone)
            with info_col3:
                st.metric("Total Vendors", len(pos_data) if not pos_data.empty else 0)

            # Detalle de compras
            st.subheader("Detalle de Compras por Droguería/Vendor")
            if not pos_data.empty:
                pos_data['porcentaje'] = (pos_data['total_compra'] / pos_data['total_compra'].sum()) * 100        
                detail_table = pos_data.copy()
                detail_table.columns = ['POS ID', 'Droguería/Vendor ID', 'Total Comprado', 'Porcentaje']
                detail_table = detail_table.round({'Porcentaje': 2})
                
                st.dataframe(
                    detail_table.style.format({
                        'Total Comprado': '${:,.2f}',
                        'Porcentaje': '{:.2f}%'
                    })
                )
                orders_pos = df_original[df_original['point_of_sale_id'] == selected_pos]
                productos_pos = df_productos_unificados[df_productos_unificados['point_of_sale_id'] == selected_pos] if 'point_of_sale_id' in df_productos_unificados.columns else pd.DataFrame()
            
                # NUEVO CÓDIGO: Mostrar tabla de vendors que son drug manufacturers
                st.subheader("Ventas de Distribuidores que son Vendors")
                if not df_vendor_dm.empty:  
                    dm_vendors_detail = crear_dataframe_vendors_dm(detail_table, df_vendor_dm)
                    if not dm_vendors_detail.empty:
                        try:
        # Obtener IDs de fabricantes (drug_manufacturer_ids)
                            dm_ids = set(df_vendor_dm['drug_manufacturer_id'].unique())
                            #st.dataframe(dm_ids)

        # Obtener la lista de drug_manufacturer_ids de la tabla de detalle
                            dm_detail_ids = set(dm_vendors_detail['Droguería/Vendor ID'].unique())
                            #st.dataframe(dm_detail_ids)
        # Convertir vendor_id a numérico para correcta comparación en orders_pos
                            orders_pos['vendor_id'] = pd.to_numeric(orders_pos['vendor_id'], errors='coerce')
        
        # Filtrar órdenes que corresponden a drug_manufacturers
                            dm_compras = orders_pos[orders_pos['vendor_id'].isin(dm_detail_ids)].copy()
                            #st.dataframe(dm_compras)
        # Total comprado a drug manufacturers
                            total_comprado_dm = dm_vendors_detail['Total Comprado'].sum()
                            #st.write(total_comprado_dm)
        # Aquí es donde calculamos las compras que son productos ganadores
        # Convertir vendor_id a numérico en productos_pos
                            productos_pos['vendor_id'] = pd.to_numeric(productos_pos['vendor_id'], errors='coerce')
        
        # Filtrar productos ganadores de este POS
                            productos_ganadores_pos = productos_pos[productos_pos['point_of_sale_id'] == selected_pos].copy()
                            #st.dataframe(productos_ganadores_pos)
        # El merge debe hacerse por super_catalog_id y point_of_sale_id, y luego verificar manualmente si vendor_id coincide
                            dm_compras_ganadores = pd.merge(
                            dm_compras, 
                            productos_ganadores_pos,
                            on=['super_catalog_id', 'point_of_sale_id'],
                            suffixes=('_comp', '_gan'),
                            how='inner'
                            ).drop_duplicates('super_catalog_id')
                            #st.write(dm_compras_ganadores['valor_vendedor_gan'].sum())

        # Verificar que los vendor_ids coincidan (compra hecha al mismo vendor que tiene mejor precio)
                            #if 'vendor_id_comp' in dm_compras_ganadores.columns: #and 'vendor_id_gan' in dm_compras_ganadores.columns:
                             #   dm_compras_ganadores = dm_compras_ganadores[
                              #  dm_compras_ganadores['vendor_id_comp'] == dm_compras_ganadores['vendor_id_gan']
                            #]

        # Calcular el valor total de compras a DMs que son productos ganadores
                            valor_dm_compras_ganadores = 0
                            if not dm_compras_ganadores.empty:
                                if 'valor_total_vendedor' in dm_compras_ganadores.columns:
                                    valor_dm_compras_ganadores = dm_compras_ganadores['valor_total_vendedor'].sum()
                                if 'unidades_pedidas' in dm_compras_ganadores.columns and 'precio_minimo' in dm_compras_ganadores.columns:
                                    valor_dm_compras_ganadores = (dm_compras_ganadores['unidades_pedidas'] * dm_compras_ganadores['precio_minimo']).sum()
                                elif 'valor_vendedor_gan' in dm_compras_ganadores.columns:
                                    valor_dm_compras_ganadores = dm_compras_ganadores['valor_vendedor_gan'].sum()
                            #st.write(valor_dm_compras_ganadores)
        # Calcular porcentaje
                            porcentaje_dm_compras_ganadores = (valor_dm_compras_ganadores / total_comprado_dm * 100) if total_comprado_dm > 0 else 0
        
        # Agregar esta información al dataframe de dm_vendors_detail
                            dm_vendors_detail['Valor Compras Ganadores'] = 0.0
                            dm_vendors_detail['% Compras Ganadores'] = 0.0
                    
                            vendor_valores = {}

        # Calcular valor para cada distribuidor específico
                            for _, row in dm_vendors_detail.iterrows():
                                dm_id = row['Droguería/Vendor ID']
            
            # Filtrar compras de este distribuidor específico que son productos ganadores
                                vendor_compras_ganadores = dm_compras_ganadores[dm_compras_ganadores['vendor_id_comp'] == dm_id] if not dm_compras_ganadores.empty else pd.DataFrame()
            
            # Calcular el valor
                                vendor_valor = 0
                                if not vendor_compras_ganadores.empty:
                                    if 'valor_total_vendedor' in vendor_compras_ganadores.columns:
                                        vendor_valor = vendor_compras_ganadores['valor_total_vendedor'].sum()
                                    elif 'valor_vendedor_gan' in vendor_compras_ganadores.columns:
                                        vendor_valor = vendor_compras_ganadores['valor_vendedor_gan'].sum()
                                    elif 'valor_vendedor_comp' in vendor_compras_ganadores.columns:
                                        vendor_valor = vendor_compras_ganadores['valor_vendedor_comp'].sum()
                
                                vendor_valores[dm_id] = vendor_valor

                            sum_vendor_valores = sum(vendor_valores.values())

                            for idx, row in dm_vendors_detail.iterrows():
                                dm_id = row['Droguería/Vendor ID']
                                vendor_valor = vendor_valores[dm_id]
                
                # Si la suma total de vendor_valores es aproximadamente igual al valor_dm_compras_ganadores,
                # usamos los valores individuales calculados
                                if abs(sum_vendor_valores - valor_dm_compras_ganadores) < 0.01 * valor_dm_compras_ganadores:  # 1% de tolerancia
                                    dm_vendors_detail.at[idx, 'Valor Compras Ganadores'] = vendor_valor
                                else:
                    # Si hay una discrepancia significativa, distribuimos el valor total proporcionalmente
                    # basado en el porcentaje de compras de cada vendor
                                    if sum_vendor_valores > 0:
                                        factor = valor_dm_compras_ganadores / sum_vendor_valores
                                        dm_vendors_detail.at[idx, 'Valor Compras Ganadores'] = vendor_valor * factor
                                    else:
                        # Si no se puede distribuir proporcionalmente, distribuir equitativamente
                                        dm_vendors_detail.at[idx, 'Valor Compras Ganadores'] = valor_dm_compras_ganadores / len(dm_vendors_detail)
                
                # Calcular el porcentaje respecto al total comprado para este vendor
                                if row['Total Comprado'] > 0:
                                    dm_vendors_detail.at[idx, '% Compras Ganadores'] = (dm_vendors_detail.at[idx, 'Valor Compras Ganadores'] / row['Total Comprado'] * 100)
            
            # Verificar que la suma de 'Valor Compras Ganadores' coincida con valor_dm_compras_ganadores
                            total_calculado = dm_vendors_detail['Valor Compras Ganadores'].sum()
                            if abs(total_calculado - valor_dm_compras_ganadores) > 0.01 * valor_dm_compras_ganadores:  # 1% de tolerancia
                                st.warning(f"Discrepancia en los cálculos: Valor DM total ({valor_dm_compras_ganadores:.2f}) ≠ Suma de valores individuales ({total_calculado:.2f})")
            


                            st.dataframe(
                            dm_vendors_detail.style.format({
                'Total Comprado': '${:,.2f}',
                'Porcentaje': '{:.2f}%',
                'Valor Compras Ganadores': '${:,.2f}',
                '% Compras Ganadores': '{:.2f}%'
                    })
                    )
        
        # Mostrar métricas de resumen
                            dm_col1, dm_col2, dm_col3 = st.columns(3)
                            with dm_col1:
                                st.metric("Total Compras a Vendors", f"${total_comprado_dm:,.2f}")
                                st.metric("% del Total de Compras", f"{(total_comprado_dm / detail_table['Total Comprado'].sum() * 100):.2f}%")
        
                            with dm_col2:
                                st.metric("Número de Vendors Drug Manufacturers", f"{len(dm_vendors_detail)}")
                                st.metric("Productos Comprados a DM que son Ganadores", f"{len(dm_compras_ganadores)}")
        
                            with dm_col3:
                                st.metric("Valor de Compras a DM que son Ganadores", f"${valor_dm_compras_ganadores:,.2f}")
                                st.metric("% de Compras a DM que son Ganadores", f"{porcentaje_dm_compras_ganadores:.2f}%")

                        except Exception as e:
                            import traceback
                    
                            st.warning(f"Error al calcular estadísticas de drug manufacturers: {str(e)}")
                            st.expander("Detalles del error", expanded=False).code(traceback.format_exc())
                    
                    else:
                        st.info("No se encontraron distribuidores que también sean fabricantes (drug manufacturers) en este punto de venta.")
                else:
                    st.warning("No se pudo cargar el archivo vendors_dm.csv o está vacío.")

            # Análisis de productos
            
            # Preparar datos de producto
        
        # 2. Filtrar productos ganadores locales que corresponden al POS seleccionado
                    
                    productos_ganadores_local_pos = productos_pos[productos_pos['point_of_sale_id'] == selected_pos]
        
                    if not productos_ganadores_local_pos.empty:
            # 3. Filtrar los que son vendidos por drug manufacturers
                        productos_ganadores_dm = productos_ganadores_local_pos[
                            productos_ganadores_local_pos['vendor_id'].isin(dm_vendor_ids)
                        ]
            
            # 4. Calcular estadísticas
                        total_productos_ganadores = len(productos_ganadores_local_pos)
                        total_valor_productos_ganadores = productos_ganadores_local_pos['valor_total_vendedor'].sum() if 'valor_total_vendedor' in productos_ganadores_local_pos.columns else 0
            
                        productos_ganadores_dm_count = len(productos_ganadores_dm)
                        valor_productos_ganadores_dm = productos_ganadores_dm['valor_total_vendedor'].sum() if not productos_ganadores_dm.empty and 'valor_total_vendedor' in productos_ganadores_dm.columns else 0
            
            # 5. Calcular proporciones
                        proporcion_productos = (productos_ganadores_dm_count / total_productos_ganadores * 100) if total_productos_ganadores > 0 else 0
                        proporcion_valor = (valor_productos_ganadores_dm / total_valor_productos_ganadores * 100) if total_valor_productos_ganadores > 0 else 0
            
            # 6. Mostrar resultados
                        #st.subheader("Proporción de Productos Ganadores vendidos por Fabricantes")
            
                        #col1, col2 = st.columns(2)
                        #with col1:
                         #   st.metric("% de Productos (Cantidad)", f"{proporcion_productos:.2f}%")
                          #  st.metric("Cantidad de Productos", f"{productos_ganadores_dm_count} de {total_productos_ganadores}")
            
                        #with col2:
                         #   st.metric("% de Valor (Dinero)", f"{proporcion_valor:.2f}%")
                          #  st.metric("Valor Total", f"${valor_productos_ganadores_dm:,.2f} de ${total_valor_productos_ganadores:,.2f}")
                    #except Exception as e:
                    #st.warning(f"Error al calcular la proporción de productos ganadores por fabricantes: {str(e)}")

            st.subheader("Análisis de Productos")

            orders_pos = df_original[df_original['point_of_sale_id'] == selected_pos]
            productos_pos = df_productos_unificados[df_productos_unificados['point_of_sale_id'] == selected_pos] if 'point_of_sale_id' in df_productos_unificados.columns else pd.DataFrame()

            # Calcular conjuntos e intersecciones
            orders_products = set(orders_pos['super_catalog_id']) if not orders_pos.empty else set()
            productos_oportunidad = set(productos_pos['super_catalog_id']) if not productos_pos.empty else set()
            
            # Calcular intersección
            intersection = pd.merge(
                productos_pos, orders_pos, 
                on=['super_catalog_id', 'point_of_sale_id'], 
                how='inner',
                suffixes=('', '_ord')
            ) if not productos_pos.empty and not orders_pos.empty else pd.DataFrame()
            
            intersection_ordenado = intersection.sort_values(['super_catalog_id', 'precio_vendedor'], ascending=[True, True])
            intersection_sin_repetidos = intersection_ordenado.drop_duplicates('super_catalog_id')
            #st.dataframe(intersection_sin_repetidos.sort_index())

            intersection_percentage = (len(intersection_sin_repetidos) / len(orders_products) * 100) if orders_products else 0
            productos_conteo= intersection['super_catalog_id'].value_counts()
            productos_repetidos = productos_conteo[productos_conteo > 1].index.tolist()
            #st.dataframe(productos_conteo)


            # Mostrar métricas de productos
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Productos en Compras Reales", f"{len(orders_products):,}")
            #with col2:
             #   st.metric("Total Productos con Oportunidad", f"{len(productos_oportunidad):,}")
            with col3:
                st.metric("Productos en Intersección no duplicados con menor precio", 
                         f"{len(intersection_sin_repetidos):,} ({intersection_percentage:.2f}%)")
                



            orders_total, products_total = 0, 0
            
            if not intersection.empty:
                # Calcular valores para productos globales
                if 'precio_total_droguería' in intersection.columns:
                    orders_total = intersection['precio_total_droguería'].sum()
                elif 'unidades_pedidas' in intersection_sin_repetidos.columns and 'precio_minimo' in intersection.columns:
                    intersection_sin_repetidos['precio_total_droguería'] = intersection_sin_repetidos['unidades_pedidas'] * intersection['precio_minimo']
                    orders_total = intersection_sin_repetidos['precio_total_droguería'].sum()
                
                if 'valor_total_vendedor' in intersection.columns:
                    products_total = intersection_sin_repetidos['valor_total_vendedor'].sum()

                # Mostrar métricas de valor
                value_col1, value_col2, value_col3 = st.columns(3)
                with value_col1:
                    st.metric("Valor en Compras Reales", f"${orders_total:,.2f}")
                with value_col2:
                    st.metric("Valor con Precios Oportunidad", f"${products_total:,.2f}")
                with value_col3:
                    # Calcular el porcentaje de ahorro
                    savings_percentage = ((orders_total - products_total) / orders_total * 100) if orders_total > 0 else 0
                    st.metric("Ahorro Potencial", f"{savings_percentage:.2f}%")


            vendor_df = pd.DataFrame()

            # Procesar datos de vendors potenciales
            vendor_analysis = []
            processed_vendors = set()

# PARTE 1: PRIMERO AÑADIR TODOS LOS VENDORS QUE SON DRUG MANUFACTURERS
# Esta es la parte crucial: asegurar que todos los vendors en dm_vendors_detail se procesen
            if not dm_vendors_detail.empty and 'Vendor Real ID' in dm_vendors_detail.columns:
                for _, row in dm_vendors_detail.iterrows():
                    if pd.notna(row.get('Vendor Real ID')):
                        vendor_id = row['Vendor Real ID']
            
            # Obtener status (si existe en vendor_pos)
                        vendor_status = obtener_status_vendor(vendor_id, selected_pos, df_vendors_pos)
                        #try:
                         #   vendor_pos_info = df_vendors_pos[
                          #      (df_vendors_pos['point_of_sale_id'] == selected_pos) & 
                           #     (df_vendors_pos['vendor_id'] == vendor_id)
                            #]
                    #        if not vendor_pos_info.empty and 'status' in vendor_pos_info.columns:
                     #           vendor_status = vendor_pos_info['status'].iloc[0]
                      #  except Exception:
    # Si hay error, continuar con status = np.nan
                       #     pass
            # Calcular valor potencial (si existe)
                        potential_value = 0
                        if not productos_pos.empty and 'vendor_id' in productos_pos.columns:
                            vendor_products = productos_pos[productos_pos['vendor_id'] == vendor_id]
                            if not vendor_products.empty and 'valor_total_vendedor' in vendor_products.columns:
                                potential_value = vendor_products['valor_total_vendedor'].sum()
            
            # Verificar si hay potencial convertido
                        valor_convertido = 0
                        if not df_potencial_convertido.empty:
                            potencial_convertido_vendor = df_potencial_convertido[
                                (df_potencial_convertido['point_of_sale_id'] == selected_pos) & 
                                (df_potencial_convertido['vendor_id'] == vendor_id)
                ]
                            if not potencial_convertido_vendor.empty:
                                valor_convertido = potencial_convertido_vendor['valor_convertido'].iloc[0]
                    
                        valor_compras_ganadores = row.get('Valor Compras Ganadores', 0)
                        if pd.notna(valor_compras_ganadores) and valor_compras_ganadores > 0:
                # Sumamos este valor al valor convertido
                            valor_convertido += valor_compras_ganadores


            # Obtener compra mínima
                        min_purchase_value = 0
                        if not df_min_purchase.empty and 'name' in df_min_purchase.columns and 'vendor_id' in df_min_purchase.columns:
                            min_purchase_info = df_min_purchase[
                                (df_min_purchase['vendor_id'] == vendor_id) & 
                                (df_min_purchase['name'] == geo_zone)
                ]
                            if not min_purchase_info.empty:
                                min_purchase_value = min_purchase_info['min_purchase'].iloc[0]
            
            # Obtener el valor comprado como DM directamente de dm_vendors_detail
                        dm_row = dm_vendors_detail[dm_vendors_detail['Vendor Real ID'] == vendor_id]
                        comprado_como_dm = dm_row['Total Comprado'].iloc[0] if not dm_row.empty else 0
            
            # Agregar a la lista de análisis
                        vendor_analysis.append({
                'Vendor ID': vendor_id,
                'Status': get_status_description(vendor_status),
                'Valor Potencial Total': potential_value,
                'Valor Convertido': valor_convertido,
                'Compra Mínima': min_purchase_value,
                # Estos valores no existían en la implementación original pero son útiles
                'Es Drug Manufacturer': 'Sí',
                'Drug Manufacturer ID': row.get('Droguería/Vendor ID'),
                'Total Comprado Como DM': comprado_como_dm
            })
            
                        processed_vendors.add(vendor_id)

            # Preparar datos para vendor analysis
        #    vendors_reales = set()
        #    if not pos_data.empty and 'Droguería/Vendor ID' in detail_table.columns:
    # Convertir a numérico para comparación adecuada
        #        detail_table['Droguería/Vendor ID'] = pd.to_numeric(detail_table['Droguería/Vendor ID'], errors='coerce')
        #        vendors_reales = set(detail_table['Droguería/Vendor ID'].unique())

# Preparar datos para vendor analysis
            if not productos_pos.empty and 'vendor_id' in productos_pos.columns:
    # Filtrar vendors válidos (con vendor_id válido)
                valid_vendors = productos_pos[productos_pos['vendor_id'].notna() & (productos_pos['vendor_id'] > 0)]
    
    # Filtrar para incluir SOLO vendors que NO aparecen en ventas reales
               # potential_vendors = valid_vendors[~valid_vendors['vendor_id'].isin(vendors_reales)]
    
    # Obtener los vendor_ids únicos de venta potencial
                unique_vendors = valid_vendors['vendor_id'].unique()

                for vendor_id in unique_vendors:
                    # Obtener status
                    if vendor_id in processed_vendors:
                        continue

                    vendor_status = obtener_status_vendor(vendor_id, selected_pos, df_vendors_pos)
                    
            #        if 'status' in valid_vendors.columns:
             #           vendor_subset = valid_vendors[valid_vendors['vendor_id'] == vendor_id]
              #          if not vendor_subset.empty:
               #             vendor_status = vendor_subset['status'].iloc[0]
                    
                    # Obtener compra mínima
                    min_purchase_value = 0
                    if not df_min_purchase.empty and 'name' in df_min_purchase.columns and 'vendor_id' in df_min_purchase.columns:
                        min_purchase_info = df_min_purchase[
                            (df_min_purchase['vendor_id'] == vendor_id) & 
                            (df_min_purchase['name'] == geo_zone)
                        ]
                        if not min_purchase_info.empty:
                            min_purchase_value = min_purchase_info['min_purchase'].iloc[0]
                    
                    # Obtener órdenes del vendor
                   # vendor_orders = orders_pos[orders_pos['vendor_id'] == vendor_id] if 'vendor_id' in orders_pos.columns else pd.DataFrame()
                    
                    # Calcular total de órdenes únicas
                    #total_orders = 0
                    #if not vendor_orders.empty and 'order_id' in vendor_orders.columns:
                     #   total_orders = len(vendor_orders['order_id'].unique())
                    
                    # Calcular órdenes que cumplen con el mínimo
                    #orders_meeting_minimum = 0
                    #if min_purchase_value > 0 and not vendor_orders.empty and 'total_compra' in vendor_orders.columns:
                     #   order_totals = vendor_orders.groupby('order_id')['total_compra'].sum()
                      #  orders_meeting_minimum = sum(order_totals >= min_purchase_value)
                    
                    # Calcular valor potencial
                    vendor_products = productos_pos[productos_pos['vendor_id'] == vendor_id]
                    potential_value = vendor_products['valor_total_vendedor'].sum() if 'valor_total_vendedor' in vendor_products.columns else 0
                    
                    # Verificar si hay potencial convertido
                    valor_convertido = 0
                    if not df_potencial_convertido.empty:
                        potencial_convertido_vendor = df_potencial_convertido[
                            (df_potencial_convertido['point_of_sale_id'] == selected_pos) & 
                            (df_potencial_convertido['vendor_id'] == vendor_id)
                        ]
                        if not potencial_convertido_vendor.empty:
                            valor_convertido = potencial_convertido_vendor['valor_convertido'].iloc[0]
                    
                    # Agregar a la lista de análisis
                    vendor_analysis.append({
                        'Vendor ID': vendor_id,
                        'Status': get_status_description(vendor_status),
                        #'Total Órdenes': total_orders,
                        #'Órdenes que Cumplen Mínimo': orders_meeting_minimum,
                        'Valor Potencial Total': potential_value,
                        'Valor Convertido': valor_convertido,
                        'Compra Mínima': min_purchase_value
                    })
                    
                    processed_vendors.add(vendor_id)
            
            # Añadir vendors que solo tienen potencial convertido
            if not df_potencial_convertido.empty:
                potencial_convertido_pos = df_potencial_convertido[df_potencial_convertido['point_of_sale_id'] == selected_pos]
                for _, row in potencial_convertido_pos.iterrows():
                    vendor_id = row['vendor_id']
                    if vendor_id in processed_vendors:
                        continue  # Ya fue procesado
                    
                    # Obtener status (si existe en vendor_pos)
                    vendor_status = obtener_status_vendor(vendor_id, selected_pos, df_vendors_pos)
        #            vendor_pos_info = df_vendors_pos[
         #               (df_vendors_pos['point_of_sale_id'] == selected_pos) & 
          #              (df_vendors_pos['vendor_id'] == vendor_id)
           #         ]
            #        if not vendor_pos_info.empty and 'status' in vendor_pos_info.columns:
             #           vendor_status = vendor_pos_info['status'].iloc[0]
                    
                    # Obtener órdenes del vendor
              #      vendor_orders = orders_pos[orders_pos['vendor_id'] == vendor_id] if 'vendor_id' in orders_pos.columns else pd.DataFrame()
                    
                    # Calcular total de órdenes únicas
                    #total_orders = 0
                    #if not vendor_orders.empty and 'order_id' in vendor_orders.columns:
                     #   total_orders = len(vendor_orders['order_id'].unique())
                    
                    # Agregar a la lista de análisis
                    vendor_analysis.append({
                        'Vendor ID': vendor_id,
                        'Status': get_status_description(vendor_status),
                      #  'Total Órdenes': total_orders,
                       # 'Órdenes que Cumplen Mínimo': 0,
                        'Valor Potencial Total': 0,
                        'Valor Convertido': row['valor_convertido'],
                        'Compra Mínima': 0
                    })
                    processed_vendors.add(vendor_id)

            # Crear DataFrame de análisis
            if vendor_analysis:
                vendor_df = pd.DataFrame(vendor_analysis)
    
    # Mostrar tabla detallada
                st.subheader("Detalle por Vendor")
    
        #        if vendor_analysis:
         #           vendor_df = pd.DataFrame(vendor_analysis)
    # En lugar del código original problemático, usar nuestra nueva función
                mostrar_tabla_vendor_detalle(vendor_df, dm_vendors_detail)
    
    # Crear gráfico
                fig = crear_grafico_oportunidades(vendor_df, df_potencial_convertido, selected_pos, dm_vendors_detail)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se encontraron vendors con venta potencial para este punto de venta.")


            # Crear resumen de oportunidades
        summary_df = create_simple_summary(productos_pos, None, orders_total, products_total, 0)
            
        if not summary_df.empty:
            st.subheader('Insights Accionables por Status de Vendor (Vista Unificada)')
    
    # Filter to get all vendors with potential value > 20000
            filtered_df = summary_df[summary_df['valor_potencial'] > 20000].copy()
    
            if not filtered_df.empty:
        # Add a readable status column
                filtered_df['Status Texto'] = filtered_df['status'].apply(lambda x: 
            "Sin Relación Comercial" if pd.isna(x) else 
            ("Activo" if x == 1 else 
             ("Pendiente" if x == 2 else 
              ("Rechazado" if x == 0 else f"Status {x}"))))
        
        # Sort by status then by potential value (descending)
                filtered_df = filtered_df.sort_values(['Status Texto', 'valor_potencial'], ascending=[True, False])
        
        # Display the unified table
                st.dataframe(filtered_df[[
            'point_of_sale_id', 'vendor_id', 'Status Texto', 'valor_potencial', 
            'tipo_oportunidad', 'porcentaje_ahorro'
        ]].style.format({
            'valor_potencial': '${:,.2f}',
            'porcentaje_ahorro': '{:.2f}%'
        }).apply(lambda x: 
            ['background-color: #ffcccb' if v == 'Sin Relación Comercial' else 
             'background-color: #90EE90' if v == 'Activo' else 
             'background-color: #FFD700' if v == 'Pendiente' else '' 
             for v in x],
            subset=['Status Texto']
        ))
        
        # Add summary metrics by status
                st.subheader('Resumen por Status')
        
        # Group by status and calculate sums and counts
                status_summary = filtered_df.groupby('Status Texto').agg({
            'valor_potencial': ['sum', 'count'],
            'porcentaje_ahorro': 'mean'
        }).reset_index()
        
        # Rename columns for clarity
                status_summary.columns = ['Status', 'Valor Potencial Total', 'Cantidad de Vendors', 'Porcentaje Ahorro Promedio']
        
        # Display the summary
                st.dataframe(status_summary.style.format({
            'Valor Potencial Total': '${:,.2f}',
            'Porcentaje Ahorro Promedio': '{:.2f}%'
        }).apply(lambda x: 
            ['background-color: #ffcccb' if v == 'Sin Relación Comercial' else 
             'background-color: #90EE90' if v == 'Activo' else 
             'background-color: #FFD700' if v == 'Pendiente' else '' 
             for v in x],
            subset=['Status']
        ))
        
        # Add a chart to visualize potential by status
                st.subheader('Distribución de Valor Potencial por Status')
        
                import plotly.express as px
        
                fig = px.pie(filtered_df, 
                     values='valor_potencial', 
                     names='Status Texto', 
                     title='Valor Potencial por Status',
                     color='Status Texto',
                     color_discrete_map={
                         'Sin Relación Comercial': '#ffcccb',
                         'Activo': '#90EE90',
                         'Pendiente': '#FFD700'
                     })
        
                fig.update_traces(textinfo='percent+label+value')
                st.plotly_chart(fig, use_container_width=True)
        
            else:
                st.info("No hay datos disponibles con valor potencial superior a $20,000.")

except Exception as e:
    st.error(f"Error al procesar los datos: {str(e)}")
    import traceback
    st.expander("Ver detalles del error", expanded=False).code(traceback.format_exc())
    st.info("Asegúrate de que todos los archivos CSV estén en el directorio correcto y tengan el formato esperado.")