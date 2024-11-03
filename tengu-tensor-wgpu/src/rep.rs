pub trait Rep {
    fn as_str() -> &'static str;
}

impl Rep for f32 {
    fn as_str() -> &'static str {
        "f32"
    }
}

impl Rep for u32 {
    fn as_str() -> &'static str {
        "u32"
    }
}

impl Rep for i32 {
    fn as_str() -> &'static str {
        "i32"
    }
}

impl Rep for bool {
    fn as_str() -> &'static str {
        "bool"
    }
}
