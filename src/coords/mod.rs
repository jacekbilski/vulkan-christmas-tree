use cgmath::num_traits::Float;
use cgmath::Point3;

/// A point P in 3-dimensional space.
/// Unlike cgmath::Point3 it uses spherical coordinates instead of cartesian.
/// The coordinate system itself is setup as in Vulkan with X axis pointing to the right, Y axis pointing downwards and Z axis pointing towards the camera so it's right-handed.
/// r is the radial (Euclidean) distance between the P and O (0, 0, 0).
/// theta (θ) is the polar angle between the positive part of Y axis and the OP line segment.
/// phi (φ) is the azimuth or azimuthal angle, an angle between the positive part of Z axis and the orthogonal projection of the line segment OP on the OXZ plane.
/// All angles are given in radians
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub struct SphericalPoint3<T> {
    pub r: T,
    pub theta: T,
    pub phi: T,
}

impl<T> SphericalPoint3<T> {
    pub fn new(r: T, theta: T, phi: T) -> Self {
        SphericalPoint3 { r, theta, phi }
    }
}

impl<T: Float> Into<Point3<T>> for SphericalPoint3<T> {
    fn into(self) -> Point3<T> {
        let x = self.r * self.theta.sin() * self.phi.sin();
        let y = self.r * self.theta.cos();
        let z = self.r * self.theta.sin() * self.phi.cos();
        Point3::new(x, y, z)
    }
}

impl<T: Float> From<Point3<T>> for SphericalPoint3<T> {
    fn from(p: Point3<T>) -> Self {
        let zero = T::from(0.).unwrap();

        let r = (p.x.powi(2) + p.y.powi(2) + p.z.powi(2)).sqrt();
        let theta = if r == zero { zero } else { (p.y / r).acos() };
        let phi = p.x.atan2(p.z);
        SphericalPoint3::new(r, theta, phi)
    }
}

/// A point P in 3-dimensional space.
/// Unlike cgmath::Point3 it uses cylindrical coordinates instead of cartesian.
/// The coordinate system itself is setup as in Vulkan with X axis pointing to the right, Y axis pointing downwards and Z axis pointing towards the camera so it's right-handed.
/// r is the radial (Euclidean) distance between the P and OY.
/// phi (φ) is the azimuth or azimuthal angle, an angle between the positive part of Z axis and the orthogonal projection of the line segment OP on the OXZ plane.
/// h or axial coordinate is the signed distance from the P to OXZ plane.
/// All angles are given in radians
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub struct CylindricalPoint3<T> {
    pub r: T,
    pub phi: T,
    pub h: T,
}

impl<T> CylindricalPoint3<T> {
    pub fn new(r: T, phi: T, h: T) -> Self {
        CylindricalPoint3 { r, phi, h }
    }
}

impl<T: Float> Into<Point3<T>> for CylindricalPoint3<T> {
    fn into(self) -> Point3<T> {
        let x = self.r * self.phi.cos();
        let y = self.h;
        let z = self.r * self.phi.sin();
        Point3::new(x, y, z)
    }
}

impl<T: Float> From<Point3<T>> for CylindricalPoint3<T> {
    fn from(p: Point3<T>) -> Self {
        let r = (p.x.powi(2) + p.z.powi(2)).sqrt();
        let phi = p.z.atan2(p.x);
        let h = p.y;
        CylindricalPoint3::new(r, phi, h)
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::FRAC_PI_2;
    use core::f32::consts::FRAC_PI_4;

    use cgmath::Point3;

    use rstest::*;

    use crate::coords::{CylindricalPoint3, SphericalPoint3};

    #[rstest(sp, expected,
    case(SphericalPoint3::new(0., 0., 0.), Point3::new(0., 0., 0.)),
    case(SphericalPoint3::new(1., 0., 0.), Point3::new(0., 1., 0.)),
    case(SphericalPoint3::new(2., 0., 0.), Point3::new(0., 2., 0.)),
    case(SphericalPoint3::new(1., FRAC_PI_2, 0.), Point3::new(0., 0., 1.)),
    case(SphericalPoint3::new(3., FRAC_PI_2, FRAC_PI_2), Point3::new(3., 0., 0.)),
    case(SphericalPoint3::new(3., FRAC_PI_4, FRAC_PI_2), Point3::new((4.5 as f32).sqrt(), (4.5 as f32).sqrt(), 0.)),
    case(SphericalPoint3::new(3., FRAC_PI_2, FRAC_PI_4), Point3::new((4.5 as f32).sqrt(), 0., (4.5 as f32).sqrt())),
    case(SphericalPoint3::new(3., FRAC_PI_4, 0.), Point3::new(0., (4.5 as f32).sqrt(), (4.5 as f32).sqrt())),
    case(SphericalPoint3::new(5., FRAC_PI_4, FRAC_PI_4), Point3::new(2.5, (12.5 as f32).sqrt(), 2.5)),
    )]
    fn spherical_point3_into_point3(sp: SphericalPoint3<f32>, expected: Point3<f32>) {
        let result: Point3<f32> = sp.into();
        let x_diff = (result.x - expected.x).abs();
        let y_diff = (result.y - expected.y).abs();
        let z_diff = (result.z - expected.z).abs();

        assert!(
            x_diff < 2. * f32::EPSILON,
            "x difference too high: {}",
            x_diff
        );
        assert!(
            y_diff < 2. * f32::EPSILON,
            "y difference too high: {}",
            y_diff
        );
        assert!(
            z_diff < 2. * f32::EPSILON,
            "z difference too high: {}",
            z_diff
        );
    }

    #[rstest(p, expected,
    case(Point3::new(0., 0., 0.), SphericalPoint3::new(0., 0., 0.)),
    case(Point3::new(0., 1., 0.), SphericalPoint3::new(1., 0., 0.)),
    case(Point3::new(0., 2., 0.), SphericalPoint3::new(2., 0., 0.)),
    case(Point3::new(0., 0., 1.), SphericalPoint3::new(1., FRAC_PI_2, 0.)),
    case(Point3::new(3., 0., 0.), SphericalPoint3::new(3., FRAC_PI_2, FRAC_PI_2)),
    case(Point3::new(3., 3., 0.), SphericalPoint3::new((18 as f32).sqrt(), FRAC_PI_4, FRAC_PI_2)),
    case(Point3::new(3., 0., 3.), SphericalPoint3::new((18 as f32).sqrt(), FRAC_PI_2, FRAC_PI_4)),
    case(Point3::new(0., 3., 3.), SphericalPoint3::new((18 as f32).sqrt(), FRAC_PI_4, 0.)),
    case(Point3::new(4., 4., 4.), SphericalPoint3::new((48 as f32).sqrt(), (4. / (48 as f32).sqrt()).acos(), FRAC_PI_4)),
    )]
    fn spherical_point3_from_point3(p: Point3<f32>, expected: SphericalPoint3<f32>) {
        let result: SphericalPoint3<f32> = SphericalPoint3::from(p);
        let r_diff = (result.r - expected.r).abs();
        let theta_diff = (result.theta - expected.theta).abs();
        let phi_diff = (result.phi - expected.phi).abs();

        assert!(
            r_diff < 2. * f32::EPSILON,
            "r difference too high: {}, expected: {}, got: {}",
            r_diff,
            expected.r,
            result.r
        );
        assert!(
            theta_diff < 2. * f32::EPSILON,
            "theta difference too high: {}, expected: {}, got: {}",
            theta_diff,
            expected.theta,
            result.theta
        );
        assert!(
            phi_diff < 2. * f32::EPSILON,
            "phi difference too high: {}, expected: {}, got: {}",
            phi_diff,
            expected.phi,
            result.phi
        );
    }

    #[rstest(cp, expected,
    case(CylindricalPoint3::new(0., 0., 0.), Point3::new(0., 0., 0.)),
    case(CylindricalPoint3::new(1., 0., 0.), Point3::new(1., 0., 0.)),
    case(CylindricalPoint3::new(2., 0., 0.), Point3::new(2., 0., 0.)),
    case(CylindricalPoint3::new(1., FRAC_PI_2, 0.), Point3::new(0., 0., 1.)),
    case(CylindricalPoint3::new(3., FRAC_PI_2, 1.), Point3::new(0., 1., 3.)),
    case(CylindricalPoint3::new(3., FRAC_PI_4, 1.), Point3::new((4.5 as f32).sqrt(), 1., (4.5 as f32).sqrt())),
    case(CylindricalPoint3::new(3., 3. * FRAC_PI_4, 0.), Point3::new(- (4.5 as f32).sqrt(), 0., (4.5 as f32).sqrt())),
    )]

    fn cylindrical_spoint3_into_point(cp: CylindricalPoint3<f32>, expected: Point3<f32>) {
        let result: Point3<f32> = cp.into();
        let x_diff = (result.x - expected.x).abs();
        let y_diff = (result.y - expected.y).abs();
        let z_diff = (result.z - expected.z).abs();

        if result != expected {
            println!(
                "Something's wrong, expected: '{:?}', got: '{:?}'",
                expected, result
            );
        }
        assert!(
            x_diff < 2. * f32::EPSILON,
            "x difference too high: {}",
            x_diff
        );
        assert!(
            y_diff < 2. * f32::EPSILON,
            "y difference too high: {}",
            y_diff
        );
        assert!(
            z_diff < 2. * f32::EPSILON,
            "z difference too high: {}",
            z_diff
        );
    }

    #[rstest(p, expected,
    case(Point3::new(0., 0., 0.), CylindricalPoint3::new(0., 0., 0.)),
    case(Point3::new(1., 0., 0.), CylindricalPoint3::new(1., 0., 0.)),
    case(Point3::new(2., 0., 0.), CylindricalPoint3::new(2., 0., 0.)),
    case(Point3::new(0., 0., 1.), CylindricalPoint3::new(1., FRAC_PI_2, 0.)),
    case(Point3::new(0., 1., 3.), CylindricalPoint3::new(3., FRAC_PI_2, 1.)),
    case(Point3::new((4.5 as f32).sqrt(), 1., (4.5 as f32).sqrt()), CylindricalPoint3::new(3., FRAC_PI_4, 1.)),
    case(Point3::new(- (4.5 as f32).sqrt(), 0., (4.5 as f32).sqrt()), CylindricalPoint3::new(3., 3. * FRAC_PI_4, 0.)),
    )]
    fn cylindrical_point3_from_point3(p: Point3<f32>, expected: CylindricalPoint3<f32>) {
        let result: CylindricalPoint3<f32> = CylindricalPoint3::from(p);
        let r_diff = (result.r - expected.r).abs();
        let phi_diff = (result.phi - expected.phi).abs();
        let h_diff = (result.h - expected.h).abs();

        if result != expected {
            println!(
                "Something's wrong, expected: '{:?}', got: '{:?}'",
                expected, result
            );
        }
        assert!(
            r_diff < 3. * f32::EPSILON,
            "r difference too high: {}, expected: {}, got: {}",
            r_diff,
            expected.r,
            result.r
        );
        assert!(
            phi_diff < 3. * f32::EPSILON,
            "phi difference too high: {}, expected: {}, got: {}",
            phi_diff,
            expected.phi,
            result.phi
        );
        assert!(
            h_diff < 3. * f32::EPSILON,
            "h difference too high: {}, expected: {}, got: {}",
            h_diff,
            expected.h,
            result.h
        );
    }
}
