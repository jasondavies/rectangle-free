#ifndef RECT_PARTITION_RESIDUAL_CENSUS_H
#define RECT_PARTITION_RESIDUAL_CENSUS_H
void residual_census_begin(void);
void residual_census_visit(const Graph*, const Poly*, const Poly*, GraphCanonWorkspace*);
void residual_census_end(RowGraphCache*, RowGraphCache*, GraphCanonWorkspace*);
#endif
