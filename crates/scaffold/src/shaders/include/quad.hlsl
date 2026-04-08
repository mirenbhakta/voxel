// Quad descriptor pack/unpack and corner generation.
//
// Quad descriptor format (25 bits used, 7 free):
//   bits  0- 4 : col       (5 bits, 0-31)
//   bits  5- 9 : row       (5 bits, 0-31)
//   bits 10-14 : layer     (5 bits, 0-31)
//   bits 15-19 : width-1   (5 bits, 0-31)
//   bits 20-24 : height-1  (5 bits, 0-31)
//   bits 25-31 : free      (7 bits, reserved)
//
// Direction is NOT encoded in the descriptor. It is implicit in the
// memory layout (direction-ordered within each chunk's quad range)
// and passed via DrawData.

#ifndef QUAD_HLSL
#define QUAD_HLSL

/// Pack quad fields into a single u32 descriptor.
uint pack_quad(uint col, uint row, uint layer, uint width, uint height) {
    return col
        | (row       << 5)
        | (layer     << 10)
        | ((width  - 1) << 15)
        | ((height - 1) << 20);
}

/// Unpacked quad fields.
struct QuadFields {
    float col;
    float row;
    float layer;
    float width;
    float height;
};

/// Unpack a u32 descriptor into float fields.
QuadFields unpack_quad(uint packed) {
    QuadFields q;
    q.col    = float(packed & 0x1F);
    q.row    = float((packed >> 5)  & 0x1F);
    q.layer  = float((packed >> 10) & 0x1F);
    q.width  = float(((packed >> 15) & 0x1F) + 1);
    q.height = float(((packed >> 20) & 0x1F) + 1);
    return q;
}

// -----------------------------------------------------------------------
// Axis vectors per direction
// -----------------------------------------------------------------------
//
// Direction indices match the Direction enum discriminants:
//   0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z
//
// Canonical face layout: layer = normal axis, col = (normal+1)%3,
// row = (normal+2)%3.

/// Outward face normal.
static const float3 NORMAL_VEC[6] = {
    float3( 1,  0,  0),  // +X
    float3(-1,  0,  0),  // -X
    float3( 0,  1,  0),  // +Y
    float3( 0, -1,  0),  // -Y
    float3( 0,  0,  1),  // +Z
    float3( 0,  0, -1),  // -Z
};

/// Column axis unit vector (col_axis = (normal+1)%3).
static const float3 COL_VEC[6] = {
    float3(0, 1, 0),  // +X -> Y
    float3(0, 1, 0),  // -X -> Y
    float3(0, 0, 1),  // +Y -> Z
    float3(0, 0, 1),  // -Y -> Z
    float3(1, 0, 0),  // +Z -> X
    float3(1, 0, 0),  // -Z -> X
};

/// Row axis unit vector (row_axis = (normal+2)%3).
static const float3 ROW_VEC[6] = {
    float3(0, 0, 1),  // +X -> Z
    float3(0, 0, 1),  // -X -> Z
    float3(1, 0, 0),  // +Y -> X
    float3(1, 0, 0),  // -Y -> X
    float3(0, 1, 0),  // +Z -> Y
    float3(0, 1, 0),  // -Z -> Y
};

/// Layer (normal) axis unit vector.
static const float3 LAYER_VEC[6] = {
    float3(1, 0, 0),  // +X
    float3(1, 0, 0),  // -X
    float3(0, 1, 0),  // +Y
    float3(0, 1, 0),  // -Y
    float3(0, 0, 1),  // +Z
    float3(0, 0, 1),  // -Z
};

/// Face offset along normal: 1.0 for positive, 0.0 for negative.
static const float IS_POSITIVE[6] = { 1, 0, 1, 0, 1, 0 };

/// Quad corner offsets for triangle strip (CCW winding for positive).
static const float2 CORNERS_POS[4] = {
    float2(0, 0),
    float2(1, 0),
    float2(0, 1),
    float2(1, 1),
};

/// Reversed winding for negative directions (outward-facing).
static const float2 CORNERS_NEG[4] = {
    float2(0, 0),
    float2(0, 1),
    float2(1, 0),
    float2(1, 1),
};

/// Select the correct corner for a given vertex index and direction.
float2 quad_corner(uint vertex_id, uint direction) {
    uint vi = vertex_id % 4;
    if (IS_POSITIVE[direction] > 0.5)
        return CORNERS_POS[vi];
    else
        return CORNERS_NEG[vi];
}

/// Reconstruct 3D local position from quad fields, corner, and direction.
float3 quad_position(QuadFields q, float2 corner, uint direction) {
    float face_col   = q.col + corner.x * q.width;
    float face_row   = q.row + corner.y * q.height;
    float face_layer = q.layer + IS_POSITIVE[direction];

    return LAYER_VEC[direction] * face_layer
         + COL_VEC[direction]   * face_col
         + ROW_VEC[direction]   * face_row;
}

#endif // QUAD_HLSL
