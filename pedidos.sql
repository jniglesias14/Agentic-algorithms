
-- Crear la tabla pedidos
CREATE TABLE IF NOT EXISTS pedidos (
    id_pedido INTEGER PRIMARY KEY,
    cliente VARCHAR(100) NOT NULL,
    producto VARCHAR(100) NOT NULL,
    cantidad INTEGER NOT NULL,
    precio_unitario DECIMAL(10, 2) NOT NULL,
    fecha_pedido DATE NOT NULL,
    estado VARCHAR(30) NOT NULL
);

-- Insertar registros de ejemplo
INSERT OR IGNORE INTO pedidos (
    id_pedido,
    cliente,
    producto,
    cantidad,
    precio_unitario,
    fecha_pedido,
    estado
)
VALUES
    (1, 'Ana García', 'Portátil', 1, 850.00, '2026-10-01', 'Entregado'),
    (2, 'Luis Martín', 'Ratón', 2, 25.00, '2026-10-02', 'Pendiente'),
    (3, 'María López', 'Teclado', 1, 45.50, '2026-10-03', 'Enviado'),
    (4, 'Pedro Sánchez', 'Monitor', 2, 210.00, '2026-10-04', 'Entregado'),
    (5, 'Ana García', 'Auriculares', 3, 35.00, '2026-10-05', 'Pendiente'),
    (6, 'Carlos Ruiz', 'Portátil', 2, 920.00, '2026-10-06', 'Enviado'),
    (7, 'Laura Pérez', 'Ratón', 4, 20.00, '2026-10-07', 'Entregado'),
    (8, 'Luis Martín', 'Monitor', 1, 199.99, '2026-10-08', 'Pendiente'),
    (9, 'Sofía Torres', 'Teclado', 2, 50.00, '2026-10-09', 'Entregado'),
    (10, 'Carlos Ruiz', 'Auriculares', 1, 59.99, '2026-10-10', 'Enviado');

-- Consultar todos los pedidos
SELECT * FROM pedidos;

-- Calcular el importe total de cada pedido
SELECT
    id_pedido,
    cliente,
    producto,
    cantidad,
    precio_unitario,
    cantidad * precio_unitario AS importe_total,
    estado
FROM pedidos;

-- Consultar pedidos pendientes
SELECT *
FROM pedidos
WHERE estado = 'Pendiente';

-- Calcular las ventas totales por cliente
SELECT
    cliente,
    SUM(cantidad * precio_unitario) AS total_comprado
FROM pedidos
GROUP BY cliente
ORDER BY total_comprado DESC;
