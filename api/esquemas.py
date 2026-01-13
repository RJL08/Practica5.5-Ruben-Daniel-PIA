from pydantic import BaseModel, Field

# clase DatosVivienda que representa los datos de entrada para la predicción del precio de la vivienda
# contiene campos como longitud, latitud, edad mediana de la vivienda, total de habitaciones,
# total de dormitorios, población, hogares, ingreso mediano y proximidad al océano
class DatosVivienda(BaseModel):
    longitude: float = Field(...)
    latitude: float = Field(...)
    housing_median_age: float = Field(..., ge=0)
    total_rooms: float = Field(..., gt=0)
    total_bedrooms: float = Field(..., gt=0)
    population: float = Field(..., gt=0)
    households: float = Field(..., gt=0)
    median_income: float = Field(..., gt=0)
    ocean_proximity: str = Field(..., description="INLAND | NEAR BAY | NEAR OCEAN | <1H OCEAN | ISLAND")

# clase RespuestaPrediccion que representa la respuesta de la API después de realizar la predicción
class RespuestaPrediccion(BaseModel):
    estado: str
    precio_predicho_regresion_lineal: float
    precio_predicho_polinomial: float
    precio_predicho_red_neuronal: float
    precio_promedio: float
