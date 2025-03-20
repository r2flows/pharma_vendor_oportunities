import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
    """
    Extrae la zona geográfica de una dirección
    
    Args:
        address: Dirección completa
        
    Returns:
        Zona geográfica extraída
    """
    partes = address.split(', ')
    return ', '.join(partes[-2:-1])

def unificar_productos_sin_duplicados(df_global, df_local):
    """
    Unifica productos con precios mínimos sin duplicados, priorizando productos locales
    
    Args:
        df_global: DataFrame con productos globales
        df_local: DataFrame con productos locales
        
    Returns:
        DataFrame unificado sin duplicados
    """
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
    """
    Carga y procesa el archivo vendors_dm.csv
    
    Returns:
        DataFrame con información de vendors que son drug manufacturers
    """
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
    
    # Obtener la lista de drug_manufacturer_ids
    dm_ids = set(df_vendor_dm['drug_manufacturer_id'].unique())
    
    # Convertir a tipo numérico para comparación adecuada
    detail_table['Droguería/Vendor ID'] = pd.to_numeric(detail_table['Droguería/Vendor ID'], errors='coerce')
    
    # Filtrar la tabla de detalle para incluir vendors donde "Droguería/Vendor ID" coincide con algún drug_manufacturer_id
    dm_vendors_detail = detail_table[detail_table['Droguería/Vendor ID'].isin(dm_ids)].copy()
    
    if not dm_vendors_detail.empty:
        # Para cada fila en dm_vendors_detail, encontrar el vendor_id correspondiente
        dm_vendors_detail['Vendor Real ID'] = None
        for idx, row in dm_vendors_detail.iterrows():
            dm_id = row['Droguería/Vendor ID']
            vendor_matches = df_vendor_dm[df_vendor_dm['drug_manufacturer_id'] == dm_id]
            if not vendor_matches.empty:
                dm_vendors_detail.at[idx, 'Vendor Real ID'] = vendor_matches.iloc[0]['vendor_id']
    
    return dm_vendors_detail

def calcular_potencial_convertido(df_pedidos, df_vendor_dm):
    """
    Calcula el potencial convertido basado en compras reales de vendors que son drug manufacturers
    
    Args:
        df_pedidos: DataFrame con información de pedidos
        df_vendor_dm: DataFrame con relaciones vendor-drug_manufacturer
        
    Returns:
        DataFrame con potencial convertido por POS y vendor
    """
    if df_pedidos.empty or df_vendor_dm.empty: 
        return pd.DataFrame()
    
    # Verificar columnas necesarias
    if 'vendor_id' not in df_pedidos.columns or 'vendor_id' not in df_vendor_dm.columns:
        print("Faltan columnas para calcular potencial convertido")
        return pd.DataFrame()
    
    # Solo incluir pedidos de vendors que son drug manufacturers
    df_pedidos_dm = pd.merge(
        df_pedidos, 
        df_vendor_dm[['vendor_id']],  # Solo necesitamos vendor_id para el merge
        on='vendor_id', 
        how='inner'
    )
    
    if df_pedidos_dm.empty:
        return pd.DataFrame()
    
    # Calcular total por point_of_sale_id y vendor_id
    if 'point_of_sale_id' in df_pedidos_dm.columns and 'vendor_id' in df_pedidos_dm.columns:
        # Calcular total_compra si no existe
        if 'total_compra' not in df_pedidos_dm.columns and 'unidades_pedidas' in df_pedidos_dm.columns and 'precio_minimo' in df_pedidos_dm.columns:
            df_pedidos_dm['total_compra'] = df_pedidos_dm['unidades_pedidas'] * df_pedidos_dm['precio_minimo']
        
        if 'total_compra' in df_pedidos_dm.columns:
            pot_convertido = df_pedidos_dm.groupby(['point_of_sale_id', 'vendor_id'])['total_compra'].sum().reset_index()
            pot_convertido.columns = ['point_of_sale_id', 'vendor_id', 'valor_convertido']
            return pot_convertido
    
    return pd.DataFrame()

def create_simple_summary(df_products, df_local_products=None, orders_total=0, products_total=0, local_products_total=0):
    """
    Crea un DataFrame resumen con la información de potencial y ahorro
    
    Args:
        df_products: DataFrame con productos
        df_local_products: DataFrame con productos locales
        orders_total: Total de órdenes
        products_total: Total de productos
        local_products_total: Total de productos locales
        
    Returns:
        DataFrame con el resumen
    """
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
    
    Args:
        vendor_df: DataFrame con información de vendors
        dm_vendors_detail: DataFrame con detalles de vendors que son drug manufacturers
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

def agregar_columna_clasificacion(df):
    """
    Agrega una columna de clasificación según las siguientes reglas:
    1. Verifica si el precio_minimo es menor que el precio_vendedor para cada producto en cada orden
    2. Si hay múltiples registros del mismo producto en una orden, identifica el que tiene el precio_vendedor más bajo
    
    Args:
        df: DataFrame con las columnas order_id, super_catalog_id, precio_minimo, precio_vendedor
        
    Returns:
        DataFrame con la nueva columna 'clasificacion'
    """
    # Crear una copia para no modificar el original
    result_df = df.copy()
    
    # Inicializar la columna de clasificación
    result_df['clasificacion'] = ""
    
    # Agrupar por order_id y super_catalog_id
    grupos = result_df.groupby(['order_id', 'super_catalog_id'])
    
    for (order_id, product_id), group in grupos:
        # Obtener el precio mínimo del producto (debe ser el mismo para todos los registros del grupo)
        precio_minimo = group['precio_minimo'].iloc[0]
        
        # Encontrar el precio_vendedor mínimo para este producto y orden
        min_precio_vendedor = group['precio_vendedor'].min()
        
        # Indices de registros del grupo
        indices = group.index
        
        for idx in indices:
            precio_vendedor = result_df.loc[idx, 'precio_vendedor']
            
            # Aplicar las reglas de clasificación
            if precio_minimo < precio_vendedor:
                result_df.loc[idx, 'clasificacion'] = "Precio droguería minimo"
            else:
                if precio_vendedor == min_precio_vendedor:
                    result_df.loc[idx, 'clasificacion'] = "Precio vendor minimo"
                else:
                    result_df.loc[idx, 'clasificacion'] = "Precio vendor no minimo"
    
    return result_df

def crear_grafico_oportunidades(vendor_df, df_potencial_convertido, selected_pos, dm_vendors_detail=None):
    """
    Crea gráfico con potencial, potencial convertido y valores comprados como DM
    
    Args:
        vendor_df: DataFrame con información de vendors
        df_potencial_convertido: DataFrame con potencial convertido
        selected_pos: ID del POS seleccionado
        dm_vendors_detail: DataFrame con detalles de vendors que son drug manufacturers
        
    Returns:
        Objeto figura de Plotly
    """
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

def actualizar_vendor_analysis(productos_pos, df_vendors_pos, orders_pos, df_potencial_convertido, 
                         dm_vendors_detail, selected_pos, geo_zone, df_min_purchase, 
                         intersection_sin_repetidos_winners):
    """
    Función principal para generar el análisis de vendors para un POS específico
    
    Args:
        productos_pos: DataFrame con productos potenciales para el POS
        df_vendors_pos: DataFrame con relaciones vendor-POS
        orders_pos: DataFrame con órdenes del POS
        df_potencial_convertido: DataFrame con potencial convertido
        dm_vendors_detail: DataFrame con detalles de vendors que son drug manufacturers
        selected_pos: ID del POS seleccionado
        geo_zone: Zona geográfica del POS
        df_min_purchase: DataFrame con información de compra mínima
        intersection_sin_repetidos_winners: DataFrame con productos ganadores (precio vendor mínimo)
        
    Returns:
        DataFrame con análisis de vendors
    """
    vendor_analysis = []
    processed_vendors = set()
    
    # Crear diccionario para almacenar los valores potenciales por vendor
    # basados en los productos ganadores (precio vendor mínimo)
    potenciales_por_vendor = {}
    
    # Calcular el potencial total por vendor basado en intersection_sin_repetidos_winners
    if not intersection_sin_repetidos_winners.empty and 'vendor_id' in intersection_sin_repetidos_winners.columns:
        for vendor_id in intersection_sin_repetidos_winners['vendor_id'].unique():
            vendor_products = intersection_sin_repetidos_winners[intersection_sin_repetidos_winners['vendor_id'] == vendor_id]
            valor_potencial = 0
            if 'precio_total_vendedor' in vendor_products.columns:
                valor_potencial = vendor_products['precio_total_vendedor'].sum()
            potenciales_por_vendor[vendor_id] = valor_potencial
    
    # PARTE 1: PRIMERO AÑADIR VENDORS QUE SON DRUG MANUFACTURERS
    if not dm_vendors_detail.empty and 'Vendor Real ID' in dm_vendors_detail.columns:
        for _, row in dm_vendors_detail.iterrows():
            if pd.notna(row.get('Vendor Real ID')):
                vendor_id = row['Vendor Real ID']
                
                # Obtener status
                vendor_status = obtener_status_vendor(vendor_id, selected_pos, df_vendors_pos)
                
                # Calcular valor potencial desde los productos ganadores
                potential_value = potenciales_por_vendor.get(vendor_id, 0)
                
                # Obtener valor DM comprado directamente de dm_vendors_detail
                comprado_como_dm = row.get('Total Comprado', 0)
                
                # Calcular valor convertido para drug manufacturers
                # Solo los drug manufacturers deben tener valores convertidos
                valor_convertido = 0
                valor_compras_ganadores = row.get('Valor Compras Ganadores', 0)
                
                if pd.notna(valor_compras_ganadores) and valor_compras_ganadores > 0:
                    # Para drug manufacturers, el valor convertido es el valor comprado como DM
                    valor_convertido = valor_compras_ganadores
                    
                    # IMPORTANTE: Restar el valor convertido del potencial para no duplicar
                    potential_value = max(0, potential_value - valor_convertido)
                
                # Obtener compra mínima
                min_purchase_value = 0
                if not df_min_purchase.empty and 'name' in df_min_purchase.columns and 'vendor_id' in df_min_purchase.columns:
                    min_purchase_info = df_min_purchase[
                        (df_min_purchase['vendor_id'] == vendor_id) & 
                        (df_min_purchase['name'] == geo_zone)
                    ]
                    if not min_purchase_info.empty:
                        min_purchase_value = min_purchase_info['min_purchase'].iloc[0]
                
                vendor_analysis.append({
                    'Vendor ID': vendor_id,
                    'Status': get_status_description(vendor_status),
                    'Valor Potencial Total': potential_value,
                    'Valor Convertido': valor_convertido,
                    'Compra Mínima': min_purchase_value,
                    'Es Drug Manufacturer': 'Sí',
                    'Drug Manufacturer ID': row.get('Droguería/Vendor ID'),
                    'Total Comprado Como DM': comprado_como_dm
                })
                
                processed_vendors.add(vendor_id)
    
    # PARTE 2: AÑADIR VENDORS REGULARES (NO DRUG MANUFACTURERS)
    # Usamos la información de intersection_sin_repetidos_winners para obtener los vendors relevantes
    if not intersection_sin_repetidos_winners.empty and 'vendor_id' in intersection_sin_repetidos_winners.columns:
        unique_vendors = intersection_sin_repetidos_winners['vendor_id'].unique()
        
        for vendor_id in unique_vendors:
            # Omitir vendors ya procesados
            if vendor_id in processed_vendors:
                continue
                
            # Obtener status
            vendor_status = obtener_status_vendor(vendor_id, selected_pos, df_vendors_pos)
            
            # Obtener compra mínima
            min_purchase_value = 0
            if not df_min_purchase.empty and 'name' in df_min_purchase.columns and 'vendor_id' in df_min_purchase.columns:
                min_purchase_info = df_min_purchase[
                    (df_min_purchase['vendor_id'] == vendor_id) & 
                    (df_min_purchase['name'] == geo_zone)
                ]
                if not min_purchase_info.empty:
                    min_purchase_value = min_purchase_info['min_purchase'].iloc[0]
            
            # Calcular valor potencial desde los productos ganadores
            potential_value = potenciales_por_vendor.get(vendor_id, 0)
            
            # Para vendors regulares (no drug manufacturers), no hay valor convertido
            valor_convertido = 0
            
            # Agregar a la lista de análisis
            vendor_analysis.append({
                'Vendor ID': vendor_id,
                'Status': get_status_description(vendor_status),
                'Valor Potencial Total': potential_value,
                'Valor Convertido': valor_convertido,  # Para vendors regulares, siempre es 0
                'Compra Mínima': min_purchase_value,
                'Es Drug Manufacturer': 'No',
                'Drug Manufacturer ID': None,
                'Total Comprado Como DM': 0
            })
            
            processed_vendors.add(vendor_id)
    
    # Crear DataFrame final
    if vendor_analysis:
        vendor_df = pd.DataFrame(vendor_analysis)
        return vendor_df
    else:
        return pd.DataFrame()
def generar_insight_simple(vendor_df, selected_pos):
    """
    Genera un DataFrame simple con relaciones POS-vendor que tienen 
    un valor potencial total superior a $20,000.
    
    Args:
        vendor_df: DataFrame con análisis de vendors
        selected_pos: ID del punto de venta seleccionado
        
    Returns:
        DataFrame con relaciones POS-vendor y valor potencial
    """
    if vendor_df.empty:
        return pd.DataFrame()
    
    # Filtrar vendors con potencial mayor a $20,000
    oportunidades_alto_valor = vendor_df[vendor_df['Valor Potencial Total'] > 20000].copy()
    
    if oportunidades_alto_valor.empty:
        return pd.DataFrame()
    
    # Crear DataFrame simplificado
    df_simple = pd.DataFrame({
        'POS ID': selected_pos,
        'Vendor ID': oportunidades_alto_valor['Vendor ID'],
        'Status': oportunidades_alto_valor['Status'],
        'Valor Potencial': oportunidades_alto_valor['Valor Potencial Total']
    })
    
    # Ordenar por Valor Potencial descendente
    df_simple = df_simple.sort_values('Valor Potencial', ascending=False)
    
    return df_simple

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
        
        # Convertir tipos de datos para cálculos correctos
        df_pedidos_proveedores_nacional['base_price'] = df_pedidos_proveedores_nacional['base_price'].astype(float)
        df_pedidos_proveedores_nacional['percentage'] = df_pedidos_proveedores_nacional['percentage'].astype(float)

        df_pedidos_proveedores_regional['base_price'] = df_pedidos_proveedores_regional['base_price'].astype(float)
        df_pedidos_proveedores_regional['percentage'] = df_pedidos_proveedores_regional['percentage'].astype(float)

        # Calcular precio_vendedor
        df_pedidos_proveedores_nacional['precio_vendedor'] = df_pedidos_proveedores_nacional['base_price'] + (df_pedidos_proveedores_nacional['base_price'] * df_pedidos_proveedores_nacional['percentage'] / 100)
        df_pedidos_proveedores_regional['precio_vendedor'] = df_pedidos_proveedores_regional['base_price'] + (df_pedidos_proveedores_regional['base_price'] * df_pedidos_proveedores_regional['percentage'] / 100)

        # Unir dataframes
        df_pedidos_proveedores = pd.concat([
            df_pedidos_proveedores_regional, df_pedidos_proveedores_nacional
        ], axis=0, ignore_index=True)
        
        # Calcular precio_total_vendedor
        if 'precio_vendedor' in df_pedidos_proveedores.columns and 'unidades_pedidas' in df_pedidos_proveedores.columns:
            df_pedidos_proveedores['precio_total_vendedor'] = (
                df_pedidos_proveedores['unidades_pedidas'].astype(float) * 
                df_pedidos_proveedores['precio_vendedor'].astype(float)
            )
        
        # Unir con relaciones vendor-pos
        if 'vendor_id' in df_pedidos_proveedores.columns and 'point_of_sale_id' in df_pedidos_proveedores.columns:
            df_pedidos_proveedores = pd.merge(
                df_pedidos_proveedores, df_vendors_pos,
                on=['point_of_sale_id', 'vendor_id'], how='left'
            )
        
        # Corregir nombres de columnas
        df_pedidos_proveedores.rename(columns={'vendor_id':'drug_manufacturer_id', 'vendor_id_y':'vendor_id'}, inplace=True)
        
        # Calcular precios mínimos locales
        cols_needed = ['point_of_sale_id', 'super_catalog_id', 'precio_minimo']
        if all(col in df_pedidos_proveedores.columns for col in cols_needed):
            min_prices = (df_pedidos_proveedores
                         .groupby(['point_of_sale_id', 'order_id','super_catalog_id'])['precio_minimo']
                         .min()
                         .reset_index())
            min_prices.columns = ['point_of_sale_id','order_id', 'super_catalog_id', 'precio_minimo_orders']
            
            # Unir para comparar precios
            df_con_precios_minimos_local = pd.merge(
                df_pedidos_proveedores, min_prices,
                on=['point_of_sale_id', 'super_catalog_id','order_id'], how='left'
            )
            
            # Clasificar productos
            df_clasificado = agregar_columna_clasificacion(df_con_precios_minimos_local)
        else:
            df_clasificado = pd.DataFrame()
        
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
        
        return pos_vendor_totals, df_pedidos, pos_order_stats, df_min_purchase, df_vendor_dm, pos_geo_zones, df_clasificado
    
    except Exception as e:
        import traceback
        print("Error en load_and_process_data:", traceback.format_exc())
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df, empty_df, empty_df, empty_df, empty_df

# Código principal
try:    
    pos_vendor_totals, df_original, pos_order_stats, df_min_purchase, df_vendor_dm, pos_geo_zones, df_clasificado = load_and_process_data()
    
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
                productos_pos = df_clasificado[df_clasificado['point_of_sale_id'] == selected_pos] if 'point_of_sale_id' in df_clasificado.columns else pd.DataFrame()
                df_vendor_winners = productos_pos[df_clasificado['clasificacion'] == 'Precio droguería minimo']
                
                # NUEVO CÓDIGO: Mostrar tabla de vendors que son drug manufacturers
                st.subheader("Ventas de Distribuidores que son Vendors")
                if not df_vendor_dm.empty:  
                    dm_vendors_detail = crear_dataframe_vendors_dm(detail_table, df_vendor_dm)
                    if not dm_vendors_detail.empty:
                        try:
                            # Obtener IDs de fabricantes (drug_manufacturer_ids)
                            dm_ids = set(df_vendor_dm['drug_manufacturer_id'].unique())

                            # Obtener la lista de drug_manufacturer_ids de la tabla de detalle
                            dm_detail_ids = set(dm_vendors_detail['Droguería/Vendor ID'].unique())
                            
                            # Filtrar órdenes que corresponden a drug_manufacturers
                            dm_compras = orders_pos[orders_pos['vendor_id'].isin(dm_detail_ids)].copy()
                            
                            # Total comprado a drug manufacturers
                            total_comprado_dm = dm_vendors_detail['Total Comprado'].sum()
                            
                            # Filtrar productos ganadores
                            productos_ganadores_pos = df_vendor_winners[productos_pos['point_of_sale_id'] == selected_pos].copy()
                            
                            # Merge para encontrar productos ganadores que son de drug manufacturers
                            dm_compras_ganadores = pd.merge(
                                dm_compras, 
                                productos_ganadores_pos,
                                on=['super_catalog_id', 'point_of_sale_id'],
                                suffixes=('_comp', '_gan'),
                                how='inner'
                            ).drop_duplicates('super_catalog_id')

                            # Calcular el valor total de compras a DMs que son productos ganadores
                            valor_dm_compras_ganadores = 0
                            if not dm_compras_ganadores.empty:
                                if 'valor_total_vendedor' in dm_compras_ganadores.columns:
                                    valor_dm_compras_ganadores = dm_compras_ganadores['valor_total_vendedor'].sum()
                                elif 'unidades_pedidas' in dm_compras_ganadores.columns and 'precio_minimo' in dm_compras_ganadores.columns:
                                    valor_dm_compras_ganadores = (dm_compras_ganadores['unidades_pedidas'] * dm_compras_ganadores['precio_minimo']).sum()
                                elif 'valor_vendedor_gan' in dm_compras_ganadores.columns:
                                    valor_dm_compras_ganadores = dm_compras_ganadores['valor_vendedor_gan'].sum()
                            
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
                st.subheader("Análisis de Productos")

                orders_pos = df_original[df_original['point_of_sale_id'] == selected_pos]
                productos_pos = df_clasificado[df_clasificado['point_of_sale_id'] == selected_pos] if 'point_of_sale_id' in df_clasificado.columns else pd.DataFrame()
               
                # Calcular conjuntos e intersecciones
                orders_products = set(orders_pos['super_catalog_id']) if not orders_pos.empty else set()
                productos_oportunidad = set(productos_pos['super_catalog_id']) if not productos_pos.empty else set()
                
                # Calcular intersección
                intersection = pd.merge(
                    productos_pos, orders_pos, 
                    on=['super_catalog_id', 'point_of_sale_id','order_id'], 
                    how='inner',
                    suffixes=('', '_ord')
                ) if not productos_pos.empty and not orders_pos.empty else pd.DataFrame()
                
                #intersection_ordenado = intersection.sort_values(['super_catalog_id', 'precio_vendedor'], ascending=[True, True])
                intersection_sin_repetidos = intersection#_ordenado.drop_duplicates('super_catalog_id')
                
                intersection_percentage = (len(intersection_sin_repetidos) / len(orders_products) * 100) if orders_products else 0
                productos_conteo= intersection['super_catalog_id'].value_counts()
                productos_repetidos = productos_conteo[productos_conteo > 1].index.tolist()

                intersection_sin_repetidos_winners = intersection_sin_repetidos[intersection_sin_repetidos['clasificacion']=='Precio vendor minimo']
                st.write(intersection_sin_repetidos_winners)
                # Mostrar métricas de productos
                #st.write(intersection_sin_repetidos_winners)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Productos en Compras Reales", f"{len(orders_products):,}")
                with col3:
                    st.metric("Productos en Intersección no duplicados con menor precio", 
                             f"{len(intersection_sin_repetidos):,} ({intersection_percentage:.2f}%)")
                    
                orders_total, products_total = 0, 0
                
                if not intersection.empty:
                    # Calcular valores para productos globales
                    if 'valor_vendedor' in intersection_sin_repetidos.columns:
                        orders_total = intersection_sin_repetidos_winners['valor_vendedor'].sum()
                    
                    if 'precio_total_vendedor' in intersection_sin_repetidos.columns:                    
                        products_total = intersection_sin_repetidos_winners['precio_total_vendedor'].sum()
                    
                    # Mostrar métricas de valor
                    value_col1, value_col2, value_col3 = st.columns(3)
                    with value_col1:
                        st.metric("Valor en Compras Reales (Potencial a Alcanzar)", f"${orders_total:,.2f}")
                    with value_col2:
                        st.metric("Valor con Precios Oportunidad", f"${products_total:,.2f}")
                    with value_col3:
                        # Calcular el porcentaje de ahorro
                        savings_percentage = ((orders_total - products_total) / orders_total * 100) if orders_total > 0 else 0
                        st.metric("Ahorro Potencial", f"{savings_percentage:.2f}%")

                # PARTE CORREGIDA: Análisis de vendors con la nueva función
                # Utilizar la función actualizar_vendor_analysis para evitar asignar valores convertidos a no-DMs
                vendor_df = actualizar_vendor_analysis(
                    productos_pos=productos_pos,
                    df_vendors_pos=df_vendors_pos,
                    orders_pos=orders_pos,
                    df_potencial_convertido=df_clasificado[df_clasificado['clasificacion'] == "Precio droguería minimo"],
                    dm_vendors_detail=dm_vendors_detail,
                    selected_pos=selected_pos,
                    geo_zone=geo_zone,
                    df_min_purchase=df_min_purchase,
                    intersection_sin_repetidos_winners=intersection_sin_repetidos_winners  # Añadir este parámetro
                )

                if not vendor_df.empty:
                    st.subheader("Detalle por Vendor")
                    
                    # Mostrar tabla detallada de vendors
                    mostrar_tabla_vendor_detalle(vendor_df, dm_vendors_detail)
                    
                    # Crear gráfico
                    fig = crear_grafico_oportunidades(vendor_df, None, selected_pos, dm_vendors_detail)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No se encontraron vendors con venta potencial para este punto de venta.")

                st.subheader("Oportunidades con Valor Potencial > $20,000")

                df_insight_simple = generar_insight_simple(vendor_df, selected_pos)

                if not df_insight_simple.empty:
    # Aplicar formato
                    styled_df = df_insight_simple.style.format({
                        'Valor Potencial': '${:,.2f}'
                    })
    
    # Aplicar colores por status
                    styled_df = styled_df.applymap(
                        lambda x: 'background-color: #90EE90' if x == "Activo" else 
                      ('background-color: #FFD700' if x == "Pendiente" else 
                     'background-color: #ffcccb' if x == "Sin Status" else ''),
                    subset=['Status']
                    )
    
    # Mostrar tabla
                    st.dataframe(styled_df)
    
    # Mostrar total
                    st.metric("Valor Potencial Total", f"${df_insight_simple['Valor Potencial'].sum():,.2f}")
                else:
                    st.info("No se encontraron oportunidades con valor potencial superior a $20,000 para este punto de venta.")
                
           
except Exception as e:
    st.error(f"Error al procesar los datos: {str(e)}")
    import traceback
    st.expander("Ver detalles del error", expanded=False).code(traceback.format_exc())
    st.info("Asegúrate de que todos los archivos CSV estén en el directorio correcto y tengan el formato esperado.")