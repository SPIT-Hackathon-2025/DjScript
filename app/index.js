import { View, Text, Image, TouchableOpacity, StyleSheet } from "react-native";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";

export default function Index() {
  const router = useRouter();

  return (
    <LinearGradient
    colors={["#1E3A8A", "#000000", ]}
      style={styles.container}
    >
      <Image
        source={require("../assets/images/image.png")}
        style={styles.image}
      />
      <Text style={styles.title}>Detect. Protect. Ride.</Text>
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          onPress={() => router.push("/signup")}
          style={[styles.button, styles.registerButton]}
        >
          <Text style={styles.buttonText}>Register</Text>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => router.push("/login")}
          style={styles.button}
        >
          <Text style={styles.buttonText}>Log In</Text>
        </TouchableOpacity>
    
     
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  image: {
    width: 320,
    height: 320,
    borderRadius: 160,
    marginBottom: 32,
    borderWidth: 4,
    borderColor: "#ffffff",
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#ffffff",
    marginBottom: 24,
    textAlign: "center",
  },
  buttonContainer: {
    flexDirection: "row",
    justifyContent: "center",
  },
  button: {
    backgroundColor: "rgba(255, 255, 255, 0.1)", // Slightly transparent white for a subtle effect
    paddingHorizontal: 32,
    paddingVertical: 12,
    borderRadius: 16,
    marginHorizontal: 8,
    borderWidth: 1,
    borderColor: "#ffffff",
  },
  registerButton: {
    marginRight: 16,
  },
  buttonText: {
    color: "#ffffff",
    fontSize: 18,
    fontWeight: "600",
  },
});
