from pydantic import BaseModel, Field


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


class RespuestaPrediccion(BaseModel):
    estado: str
    precio_predicho_regresion_lineal: float
    precio_predicho_polinomial: float
    precio_predicho_red_neuronal: float
    precio_promedio: float
