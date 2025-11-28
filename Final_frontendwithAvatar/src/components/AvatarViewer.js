import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useFBX } from '@react-three/drei';
import { Suspense, useState, useEffect, useRef } from 'react';
import { AnimationMixer } from 'three';
import backgroundImage from './image.jpeg';

// Custom Button component
function Button({ children, onClick, style, ...props }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '10px 20px',
        backgroundColor: '#007BFF',
        color: '#fff',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
        ...style,
      }}
      {...props}
    >
      {children}
    </button>
  );
}

function Avatar({ animationName }) {
  // Preload all animations
  const idle = useFBX('/Idle.fbx');
  const talking = useFBX('/Talking.fbx');
  const writing = useFBX('/Writing.fbx');

  // Use the idle model as the base
  const model = useRef(idle);
  const mixer = useRef(new AnimationMixer(model.current));
  const actions = useRef({});

  // Set up actions for each animation once they're loaded
  useEffect(() => {
    if (idle.animations.length && talking.animations.length && writing.animations.length) {
      actions.current.idle = mixer.current.clipAction(idle.animations[0]);
      actions.current.talking = mixer.current.clipAction(talking.animations[0]);
      actions.current.writing = mixer.current.clipAction(writing.animations[0]);

      // Start with the idle animation
      actions.current.idle.play();
    }
    return () => mixer.current.stopAllAction();
  }, [idle, talking, writing]);

  // Change animation when animationName prop updates
  useEffect(() => {
    const newAction = actions.current[animationName];
    if (newAction) {
      // Crossfade from any currently playing actions to the new action
      Object.values(actions.current).forEach((action) => {
        if (action !== newAction) {
          action.crossFadeTo(newAction, 0.5, false);
        }
      });
      newAction.reset().play();
    }
  }, [animationName]);

  useFrame((_, delta) => {
    mixer.current.update(delta);
  });

  return (
    <primitive
      object={model.current}
      scale={0.2}
      position={[0, animationName === 'writing' ? -20 : -25, 0]}
    />
  );
}

export default function AvatarViewer() {
  const animationSequence = [
    { name: 'idle', duration: 5000 },
    { name: 'talking', duration: 10000 },
    { name: 'writing', duration: 7000 },
  ];
  const [index, setIndex] = useState(0);

  // Optional manual controls using buttons
  const goToPrevious = () => {
    setIndex((prevIndex) =>
      prevIndex === 0 ? animationSequence.length - 1 : prevIndex - 1
    );
  };

  const goToNext = () => {
    setIndex((prevIndex) => (prevIndex + 1) % animationSequence.length);
  };

  // Automated transition based on duration
  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prevIndex) => (prevIndex + 1) % animationSequence.length);
    }, animationSequence[index].duration);
    return () => clearInterval(interval);
  }, [index, animationSequence]);

  return (
    <div className="flex flex-col items-center w-full h-screen bg-black p-4">
      <h1 className="myheader">
        <center>MedSurance</center>
      </h1>

      <div style={{
        width: '100%',
        height: '80vh',
        position: 'relative',
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
      }}>
        <Canvas
          style={{ width: '100%', height: '100%' }}
          camera={{ position: [0, 5, 15], fov: 45 }}
        >
          <ambientLight intensity={1.5} />
          <directionalLight position={[5, 5, 5]} intensity={1.5} />
          <Suspense fallback={null}>
            <Avatar animationName={animationSequence[index].name} />
          </Suspense>
          <OrbitControls
            enableZoom
            enableRotate
            minDistance={30}
            maxDistance={30}
          />
        </Canvas>
      </div>
      <div className="mt-4 flex gap-4">
      </div>
    </div>
  );
}
